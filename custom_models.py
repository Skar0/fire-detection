import imghdr
import os
import math
import numpy as np
from keras.preprocessing import image
from matplotlib import pyplot as plt
from keras import Model
from keras.applications import InceptionV3
from keras.applications.inception_v3 import preprocess_input as inception_preprocess_input
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Dense
from keras.utils import np_utils
from keras_preprocessing.image import ImageDataGenerator

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
    display = False
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
                # converting the loaded images to numpy arrays
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
                    print("------:-:-:-:-:--::---")
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

    train_labels, val_labels = np.empty([1, 0]), np.empty([1, 0])
    train_samples, val_samples = np.empty([1, 0]), np.empty([1, 0])

    for class_name in os.listdir(dataset_path):
        # putting images paths and labels in lists to work on them
        images_paths, labels = [], []

        class_path = os.path.join(dataset_path, class_name)
        class_id = classes_names.index(class_name)  # class id = index of the class_name in classes_name, later o-h enc

        # here we are considering all paths for images labeled class_id
        for path in os.listdir(class_path):
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


def train_and_save_Inception_based_model(dataset_path, percentage=0.9, nbr_epochs=10, batch_size=32):
    """
    :param dataset_path:
    :param percentage: percentage of samples to be used for training. Must be in [0,1].
    :param nbr_epochs:
    :param batch_size:
    """

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
