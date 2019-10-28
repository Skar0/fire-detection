import argparse
import imghdr
import os
import math
import numpy as np
from keras.engine.saving import load_model
from keras.optimizers import SGD
from keras.preprocessing import image
from matplotlib import pyplot as plt
from keras import Model
from keras.applications import InceptionV3
from keras.applications.inception_v3 import preprocess_input as inception_preprocess_input
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Dense
from keras.utils import np_utils
from keras_preprocessing.image import ImageDataGenerator

from video_fire_detection import detect_fire_on_the_fly

"""
This module contains functions used to create training and validation datasets using with proper representation of each
class. It also contains a batch generator which performs data augmentation (shifts, rotations, flips and zooms) on the
fly. Finally, transfer learning from an InceptionV3-based model is performed and the model is re-trained for fire
images using our augmented dataset.
"""

classes = ['fire', 'no_fire', 'start_fire']
nbr_classes = 3


def generate_from_paths_and_labels(images_paths, labels, batch_size, preprocessing, augment, image_size=(224, 224)):
    """
    Generator to give to the fit function, generates batches of samples for training.
    This avoids to load the full dataset in memory. This can also be a Keras class.
    Applies random transformation to images.

    An alternative to this method is to create a flow() generator for each batch and apply transformations that way, or
    use flow_from_directory (but then we have to remove validation data from the directories which is annoying). This
    method applies the transformations randomly on each batch using the specific function of ImageDataGenerator.
    Prepocessing is then applied on the batch manually (it is done automatically with flow generators if a preprocessing
    function is given to ImageDataGenerator.
    :param images_paths:
    :param labels:
    :param batch_size:
    :param preprocessing:
    :param augment: whether to augment the data.
    :param image_size:
    :return:
    """

    display = False  # whether to display data augmentation on a subset of the batch

    number_samples = len(images_paths)
    if augment:
        datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True,
                                     rotation_range=30,
                                     brightness_range=[0.7, 1.3], zoom_range=[0.7, 1.3])
    while 1:
        perm = np.random.permutation(number_samples)  # randomize the order of the images (to be done after each epoch)

        # apply the permutations
        images_paths = images_paths[perm]
        labels = labels[perm]

        # from 0 to number_samples by batch_size increment to generate batches
        # this assumes there are number_samples / batch_size batches in an epoch
        # which ensures that each samples is only fed once to the network at each epoch
        for i in range(0, number_samples, batch_size):
            # a batch is a list of image paths : images_paths[i:i + batch_size]
            # map transforms all paths to images using keras.preprocessing.image
            inputs = list(map(
                lambda x: image.load_img(x, target_size=image_size),
                images_paths[i:i + batch_size]
            ))

            if augment:
                # converting the loaded images to numpy arrays and applying augmentation
                inputs = np.array(list(map(
                    lambda x: datagen.random_transform(image.img_to_array(x)),
                    inputs
                )))
            else:
                # converting the loaded images to numpy arrays
                inputs = np.array(list(map(
                    lambda x: image.img_to_array(x),
                    inputs
                )))

            if display:
                for j in range(9):
                    # define subplot
                    plt.subplot(330 + 1 + j)
                    # generate batch of images
                    # convert to unsigned integers for viewing
                    img = inputs[j].astype('uint8')
                    # plot raw pixel data
                    plt.imshow(img)
                    print("class")
                    print(labels[j])

            # preprocessing the batch might notably normalize between 0 and 1 the RGB values, this is model-dependant
            inputs = preprocessing(inputs)

            # yields the image batch and corresponding labels
            yield (inputs, labels[i:i + batch_size])


def extract_dataset(dataset_path, classes_names, percentage):
    """
    Assumes that dataset_path/classes_names[0] is a folder containing all images of class classes_names[0].
    All image paths are loaded into a numpy array, corresponding labels are one-hot encoded and put into a numpy array.

    The validation dataset is composed of 1-percentage % of the images of each class in order to ensure that the
    validation dataset is representative of the data and prevent skewed class representation in validation dataset.

    Training and validation datasets are then composed of the correct percentage of each class and thus of the whole
    dataset. They are then shuffled to prevent problems since they are composed of samples loaded in order of their
    class.
    :param dataset_path: path to the root of the dataset.
    :param classes_names: names of the classes.
    :param percentage: percentage of samples to be used for training, the rest is for validation. Must be in [0,1].
    :return: (x_train, y_train), (x_val, y_val) a list of image paths and a list of corresponding labels for training
    and validation.
    """

    num_classes = len(classes_names)

    # ignore hidden files
    def listdir_nohidden(path):
        for f in os.listdir(path):
            if not f.startswith('.'):
                yield f

    train_labels, val_labels = np.empty([1, 0]), np.empty([1, 0])
    train_samples, val_samples = np.empty([1, 0]), np.empty([1, 0])

    for class_name in listdir_nohidden(dataset_path):
        # putting images paths and labels in lists to work on them
        images_paths, labels = [], []

        class_path = os.path.join(dataset_path, class_name)
        class_id = classes_names.index(class_name)  # class id = index of the class_name in classes_name, later o-h enc

        # here we are considering all paths for images labeled class_id
        for path in listdir_nohidden(class_path):
            path = os.path.join(class_path, path)  # image path
            # test the image data contained in the file , and returns a string describing the image type
            if imghdr.what(path) is None:
                # this is not an image file
                continue
            images_paths.append(path)
            labels.append(class_id)
        # print(class_name)
        # print(len(labels))
        # here all samples of class_name are in images_paths, labels
        # we now shuffle the samples and select percentage

        # one-hot encode the labels
        labels_oh = np.array(labels)
        # convert images_paths to numpy array to apply permutation
        images_paths = np.array(images_paths)

        number_samples = len(images_paths)
        perm = np.random.permutation(number_samples)
        labels_oh = labels_oh[perm]
        images_paths = images_paths[perm]

        # percentage % of samples used for training
        border = math.floor(percentage * len(images_paths))

        train_labels_temp, val_labels_temp = labels_oh[:border], labels_oh[border:]
        train_samples_temp, val_samples_temp = images_paths[:border], images_paths[border:]

        train_labels = np.append(train_labels, train_labels_temp)
        val_labels = np.append(val_labels, val_labels_temp)

        train_samples = np.append(train_samples, train_samples_temp)
        val_samples = np.append(val_samples, val_samples_temp)

        # print(len(train_samples))
        # print(len(val_samples))

    number_samples_train = len(train_samples)
    perm = np.random.permutation(number_samples_train)
    train_labels = np_utils.to_categorical(train_labels, num_classes)
    train_labels = train_labels[perm]
    train_samples = train_samples[perm]

    number_samples_val = len(val_samples)
    perm = np.random.permutation(number_samples_val)
    val_labels = np_utils.to_categorical(val_labels, num_classes)
    val_labels = val_labels[perm]
    val_samples = val_samples[perm]

    print("Training on %d samples" % number_samples_train)
    print("Validation on %d samples" % number_samples_val)

    return (train_samples, train_labels), (val_samples, val_labels)


def create_Inception_based_model():
    """
    Inception-based model.
    :return: the model.
    """
    # weights are pre-trained with imagenet
    base_model = InceptionV3(include_top=False, weights='imagenet', pooling='max', input_shape=(224, 224, 3))

    x = base_model.output
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(nbr_classes, activation='softmax')(x)  # dense layer with neurons with softmax
    model = Model(inputs=base_model.inputs, outputs=predictions)  # input is based model input, output is custom

    # we set every layer to be trainable
    for layer in model.layers:
        layer.trainable = True

    return model


def train_and_save_Inception_based_model(dataset_path, fine_tune_existing=None, learning_rate=0.001, percentage=0.9, nbr_epochs=10, batch_size=32):
    """
    Creates and train an InceptionV3-based model on the fire images dataset or fine-tunes an pre-trained model with a
    custom learning rate.
    :param dataset_path: path to the dataset.
    :param fine_tune_existing: whether a model was already trained and to just continue fine-tuning it. Its value should
    be the path to the existing model to be loaded or None if no prior model is to be loaded.
    :param learning_rate: when fine-tuning, the learning rate can be specified.
    :param percentage: percentage of samples to be used for training. Must be in [0,1].
    :param nbr_epochs:
    :param batch_size:
    """

    #  if a pre-trained model is specified, load it. Else create the model.
    if fine_tune_existing is not None:
        Inception_based_model = load_model(fine_tune_existing)
    else:
        Inception_based_model = create_Inception_based_model()

    Inception_based_model_save_folder = "model-saves/Inception_based/"

    # create save path
    if not os.path.exists(Inception_based_model_save_folder):
        os.makedirs(Inception_based_model_save_folder)

    Inception_based_model_save_path = Inception_based_model_save_folder + "best_trained_save.h5"

    # checkpoints

    # We can do learning rate adaptation later as part of fine tuning or use adaptive optimizer (rmsprop, adam)
    # keras.callbacks.callbacks.LearningRateScheduler(schedule, verbose=0)
    # keras.callbacks.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto',
    # min_delta=0.0001, cooldown=0, min_lr=0)

    # saves the model when validation accuracy improves
    save_on_improve = ModelCheckpoint(Inception_based_model_save_path, monitor='val_acc', verbose=1,
                                      save_best_only=True, save_weights_only=False, mode='max')

    # EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto',baseline=None, res
    # tore_best_weights=False)

    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False,
                              write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                              embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

    cb = [save_on_improve, tensorboard]

    # loss is categorical since we are classifying
    # if a pre-trained model was specified, we are fine tuning and need to take the custom learning rate into account
    if fine_tune_existing is not None:

        sgd = SGD(lr=learning_rate, momentum=0.0, nesterov=False)  # default is 0.01
        Inception_based_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    else:
        Inception_based_model.compile(loss='categorical_crossentropy', optimizer="sgd", metrics=['accuracy'])

    (train_samples, train_labels), (val_samples, val_labels) = extract_dataset(dataset_path, classes, percentage)

    training_sample_generator = generate_from_paths_and_labels(train_samples, train_labels, batch_size,
                                                               inception_preprocess_input, True,
                                                               image_size=(224, 224, 3))

    validation_sample_generator = generate_from_paths_and_labels(val_samples, val_labels, batch_size,
                                                                 inception_preprocess_input, False,
                                                                 image_size=(224, 224, 3))

    nbr_train_samples = len(train_samples)
    nbr_val_samples = len(val_samples)

    # call to fit using a generator
    history = Inception_based_model.fit_generator(
        generator=training_sample_generator,
        steps_per_epoch=math.ceil(nbr_train_samples / batch_size),
        epochs=nbr_epochs,
        validation_data=validation_sample_generator,
        validation_steps=math.ceil(nbr_val_samples / batch_size),
        callbacks=cb, verbose=1)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Convolutional neural network for forest fire detection',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    subparsers = parser.add_subparsers(title='Mode selection',
                                       description='Network can be trained on a provided dataset or predictions can be'
                                                   'made using a pre-trained model',
                                       help='', dest='mode')

    subparsers.required = True

    parser_train = subparsers.add_parser('train',
                                         help='Create and train the InceptionV3-based model.',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser_train.add_argument('-data',
                              type=str,
                              action='store',
                              dest='dataset',
                              help='Path to the dataset on which to train.',
                              default=argparse.SUPPRESS,
                              required=True)

    parser_train.add_argument('-prop',
                              type=float,
                              action='store',
                              dest='proportion',
                              help='Proportion of the dataset to be used for training (the rest is for validation).',
                              default=argparse.SUPPRESS,
                              required=True)

    parser_train.add_argument('-epochs',
                              type=int,
                              action='store',
                              dest='epochs',
                              help='Number of epochs.',
                              default=10,
                              required=False)

    parser_train.add_argument('-batch',
                              type=int,
                              action='store',
                              dest='batch_size',
                              help='Size of a batch.',
                              default=32,
                              required=False)

    parser_tune = subparsers.add_parser('tune', help='Fine-tune an pre-trained Inception-V3-based model.',
                                        formatter_class = argparse.ArgumentDefaultsHelpFormatter)

    parser_tune.add_argument('-model',
                             type=str,
                             action='store',
                             dest='model_path',
                             help='Path to the pre-trained model.',
                             default=argparse.SUPPRESS,
                             required=True)

    parser_tune.add_argument('-lr',
                             type=float,
                             action='store',
                             dest='learning_rate',
                             help='Learning rate to be used for fine-tuning.',
                             default=0.001,
                             required=False)

    parser_tune.add_argument('-data',
                              type=str,
                              action='store',
                              dest='dataset',
                              help='Path to the dataset on which to train.',
                              default=argparse.SUPPRESS,
                              required=True)

    parser_tune.add_argument('-prop',
                              type=float,
                              action='store',
                              dest='proportion',
                              help='Proportion of the dataset to be used for training (the rest is for validation).',
                              default=argparse.SUPPRESS,
                              required=True)

    parser_tune.add_argument('-epochs',
                              type=int,
                              action='store',
                              dest='epochs',
                              help='Number of epochs.',
                              default=10,
                              required=False)

    parser_tune.add_argument('-batch',
                              type=int,
                              action='store',
                              dest='batch_size',
                              help='Size of a batch.',
                              default=32,
                              required=False)

    parser_predict = subparsers.add_parser('predict',
                                           help='Perform prediction on a provided picture.')

    parser_predict.add_argument('-path',
                                type=str,
                                action='store',
                                dest='image_path',
                                help='Path to an image.',
                                default=argparse.SUPPRESS,
                                required=True)

    parser_predict.add_argument('-model',
                                type=str,
                                action='store',
                                dest='model_path',
                                help='Path to a trained model.',
                                default=argparse.SUPPRESS,
                                required=True)

    parser_video = subparsers.add_parser('video',
                                           help='Perform prediction on a video.',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser_video.add_argument('-in',
                                type=str,
                                action='store',
                                dest='input_video_path',
                                help='Path to an mp4 video.',
                                default=argparse.SUPPRESS,
                                required=True)

    parser_video.add_argument('-out',
                                type=str,
                                action='store',
                                dest='output_video_path',
                                help='Path to output annotated mp4 video.',
                                default=argparse.SUPPRESS,
                                required=True)

    parser_video.add_argument('-model',
                                type=str,
                                action='store',
                                dest='model_path',
                                help='Path to a trained model.',
                                default=argparse.SUPPRESS,
                                required=True)

    parser_video.add_argument('-freq',
                                type=str,
                                action='store',
                                dest='freq',
                                help='Prediction is to be made every freq frames.',
                                default=12,
                                required=False)

    parser_test = subparsers.add_parser('test',
                                           help='Test a model on a test set of images.')

    parser_test.add_argument('-path',
                                type=str,
                                action='store',
                                dest='test_set_path',
                                help='Path to a test set.',
                                default=argparse.SUPPRESS,
                                required=True)

    parser_test.add_argument('-model',
                                type=str,
                                action='store',
                                dest='model_path',
                                help='Path to a trained model.',
                                default=argparse.SUPPRESS,
                                required=True)

    parsed = parser.parse_args()

    if parsed.mode == "train":

        train_and_save_Inception_based_model(parsed.dataset,
                                             fine_tune_existing=None,
                                             learning_rate=0.001,
                                             percentage=parsed.proportion,
                                             nbr_epochs=parsed.epochs,
                                             batch_size=parsed.batch_size)

    elif parsed.mode == "tune":

        train_and_save_Inception_based_model(parsed.dataset,
                                             fine_tune_existing=parsed.model_path,
                                             learning_rate=parsed.learning_rate,
                                             percentage=parsed.proportion,
                                             nbr_epochs=parsed.epochs,
                                             batch_size=parsed.batch_size)
    elif parsed.mode == "predict":
        print('image path: ', parsed.image_path)
        print('model path: ', parsed.model_path)

    elif parsed.mode == "video":

        detect_fire_on_the_fly(parsed.input_video_path,
                               parsed.output_video_path,
                               parsed.model_path,
                               inception_preprocess_input,
                               (224,224),
                               parsed.freq)

    elif parsed.mode == "test":
        print('test path: ', parsed.test_set_path)
        print('model path: ', parsed.model_path)
