import imghdr
import os
import math
import numpy as np
from keras.models import load_model
from keras.optimizers import SGD
from keras.preprocessing import image
from matplotlib import pyplot as plt
from keras import Model
from keras.applications import InceptionV3
from keras.applications.inception_v3 import preprocess_input as inception_preprocess_input
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Dense
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

"""
This module contains functions used to create training and validation datasets using with proper representation of each
class. It also contains a batch generator which performs data augmentation (shifts, rotations, flips and zooms) on the
fly. Finally, transfer learning from an InceptionV3-based model is performed and the model is re-trained for fire
images using our augmented dataset.
"""

# we work with three classes for this whole project
classes = ['fire', 'no_fire', 'start_fire']
nbr_classes = 3


def augmented_batch_generator(images_paths, labels, batch_size, preprocessing, augment, image_size=(224, 224)):
    """
    Generator to give to the fit function, generates batches of samples for training. This avoids to load the full
    dataset in memory and works from lists of paths to the actual samples. This can also be implemented as a class.
    Random transformations (shifts, rotations, flips and zooms) can be applied to images for data augmentation.

    An alternative to this method is to create a flow() generator for each batch and apply transformations that way, or
    use flow_from_directory (but then we have to remove validation data from the directories which is annoying). This
    method applies the transformations randomly on each batch using the dedicated function of ImageDataGenerator.
    Prepocessing is then applied on the batch manually (it is done automatically with flow generators if a preprocessing
    function is given to ImageDataGenerator).

    :param images_paths: list of paths to images.
    :param labels: list of corresponding labels for the images.
    :param batch_size: size of the generated batches.
    :param preprocessing: preprocessing to be applied to the images, as required by the network (usually normalisation
    in [0, 1] of pixel values and so on).
    :param augment: whether to augment the data by applying random transformations.
    :param image_size: size for the generated images, default is 224x224.
    """

    display = False  # whether to display data augmentation on a subset of the batch (for debugging purposes)

    number_samples = len(images_paths)  # number of images

    # if data is to be augmented, create a ImageDataGenerator object to apply the transformations
    if augment:
        data_transformer = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True,
                                              rotation_range=20, brightness_range=[0.7, 1.3], zoom_range=[0.8, 1.3])

    # infinite loop for the generator
    while 1:
        perm = np.random.permutation(number_samples)  # randomize the order of the images (done after each epoch)

        # apply the permutations to images and labels
        images_paths = images_paths[perm]
        labels = labels[perm]

        # loop from 0 to number_samples by batch_size increment to generate batches
        # this assumes there are number_samples / batch_size batches in an epoch
        # which ensures that each samples is only fed once to the network at each epoch
        for i in range(0, number_samples, batch_size):

            # a batch is a list of image paths : images_paths[i:i + batch_size]
            # map transforms all paths to images using keras.preprocessing.image
            batch = list(map(
                lambda x: image.load_img(x, target_size=image_size),
                images_paths[i:i + batch_size]
            ))

            if augment:
                # converting the loaded images to numpy arrays and applying augmentation
                batch = np.array(list(map(
                    lambda x: data_transformer.random_transform(image.img_to_array(x)),
                    batch
                )))
            else:
                # converting the loaded images to numpy arrays
                batch = np.array(list(map(
                    lambda x: image.img_to_array(x),
                    batch
                )))

            if display:
                for j in range(9):
                    plt.subplot(330 + 1 + j)
                    img = batch[j].astype('uint8')
                    plt.imshow(img)
                    print(labels[j])

            # preprocessing the batch might notably normalize between 0 and 1 the RGB values, this is model-dependant
            batch = preprocessing(batch)

            # yields the image batch and corresponding labels
            yield (batch, labels[i:i + batch_size])


def extract_dataset(dataset_path, classes_names, percentage):
    """
    Assumes that dataset_path/classes_names[0] is a folder containing all images of class classes_names[0] and this for
    all classes. All image paths are loaded into a numpy array, corresponding labels are one-hot encoded and put into a
    numpy array. This is later fed to a batch generator which avoids loading the whole set of images in memory.

    The validation dataset is composed of (1 - percentage)% of the images of each class in order to ensure that the
    validation dataset is representative of the data and prevent skewed class representation in validation dataset.

    Training and validation datasets are composed of the correct percentage of each class and thus of the whole
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

    # initialize the arrays
    train_labels, val_labels = np.empty([1, 0]), np.empty([1, 0])
    train_samples, val_samples = np.empty([1, 0]), np.empty([1, 0])

    for class_name in listdir_nohidden(dataset_path):
        # putting images paths and labels in lists to work on them
        images_paths, labels = [], []

        class_path = os.path.join(dataset_path, class_name)
        class_id = classes_names.index(class_name)  # class id = index of the class_name in classes_name, later o-h enc

        # here we consider all paths for images labeled class_id
        for path in listdir_nohidden(class_path):
            path = os.path.join(class_path, path)  # image path
            # test the image data contained in the file , and returns a string describing the image type
            if imghdr.what(path) is None:
                # this is not an image file
                continue
            images_paths.append(path)
            labels.append(class_id)

        # here all samples of class_name are in images_paths and the corresponding labels in labels
        # we now shuffle the samples and select the correct percentage for training percentage

        # one-hot encode the labels
        labels_oh = np.array(labels)
        # convert images_paths to numpy array to apply permutation
        images_paths = np.array(images_paths)

        # perform permutation
        number_samples = len(images_paths)
        perm = np.random.permutation(number_samples)
        labels_oh = labels_oh[perm]
        images_paths = images_paths[perm]

        # percentage % of samples used for training
        border = math.floor(percentage * len(images_paths))

        # select the correct percentage of samples
        train_labels_temp, val_labels_temp = labels_oh[:border], labels_oh[border:]
        train_samples_temp, val_samples_temp = images_paths[:border], images_paths[border:]

        train_labels = np.append(train_labels, train_labels_temp)
        val_labels = np.append(val_labels, val_labels_temp)

        train_samples = np.append(train_samples, train_samples_temp)
        val_samples = np.append(val_samples, val_samples_temp)

    # apply permutation to the training and validation sets since they are created in order of their labels
    # not doing this may later lead to batches containing only one class
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


def create_inception_based_model():
    """
    Inception-based model, uses InceptionV3 network without top layer and using max global pooling. Custom top layer
    is added to perform classification.

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


def train_inception_based_model(dataset_path,
                                fine_tune_existing=None,
                                learning_rate=0.001,
                                percentage=0.9,
                                nbr_epochs=10,
                                batch_size=32):
    """
    Creates and train an InceptionV3-based model on the fire images dataset or fine-tunes an pre-trained model with a
    custom learning rate.

    :param dataset_path: path to the dataset.
    :param fine_tune_existing: whether a model was already trained and to just continue fine-tuning it. Its value should
    be the path to the existing model which needs to be loaded or None if no prior model is to be loaded.
    :param learning_rate: when fine-tuning, the learning rate can be specified.
    :param percentage: percentage of samples to be used for training. Must be in [0,1].
    :param nbr_epochs: number of epochs.
    :param batch_size: batch size.
    """

    #  if a pre-trained model is specified, load it. Else create the model.
    if fine_tune_existing is not None:
        inception_based_model = load_model(fine_tune_existing)
    else:
        inception_based_model = create_inception_based_model()

    inception_based_model_save_folder = "model-saves/Inception_based/"

    # create save folder
    if not os.path.exists(inception_based_model_save_folder):
        os.makedirs(inception_based_model_save_folder)

    inception_based_model_save_path = inception_based_model_save_folder + "best_trained_save.h5"

    # We can do learning rate adaptation later as part of fine tuning or use adaptive optimizer (rmsprop, adam)
    # keras.callbacks.callbacks.LearningRateScheduler(schedule, verbose=0)
    # keras.callbacks.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto',
    # min_delta=0.0001, cooldown=0, min_lr=0)
    # EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto',baseline=None, res
    # tore_best_weights=False)

    # saves the model when validation accuracy improves, overwrites previously saved model
    save_on_improve = ModelCheckpoint(inception_based_model_save_path,
                                      monitor='val_acc',
                                      verbose=1,
                                      save_best_only=True,
                                      save_weights_only=False,
                                      mode='max')

    # write logs to ./logs for Tensorboard visualization
    tensorboard = TensorBoard(log_dir='./logs',
                              histogram_freq=0,
                              batch_size=32,
                              write_graph=True,
                              write_grads=False,
                              write_images=False,
                              embeddings_freq=0,
                              embeddings_layer_names=None,
                              embeddings_metadata=None,
                              embeddings_data=None,
                              update_freq='epoch')

    # callbacks
    cb = [save_on_improve, tensorboard]

    # loss is categorical since we are classifying
    # if a pre-trained model was specified, we are fine tuning and need to take the custom learning rate into account
    if fine_tune_existing is not None:
        sgd = SGD(lr=learning_rate, momentum=0.0, nesterov=False)  # default lr is 0.01
        inception_based_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    else:
        inception_based_model.compile(loss='categorical_crossentropy', optimizer="sgd", metrics=['accuracy'])

    # extract the image paths to give to the generator
    (train_samples, train_labels), (val_samples, val_labels) = extract_dataset(dataset_path, classes, percentage)

    training_sample_generator = augmented_batch_generator(train_samples,
                                                          train_labels,
                                                          batch_size,
                                                          inception_preprocess_input,
                                                          augment=True,
                                                          image_size=(224, 224, 3))

    validation_sample_generator = augmented_batch_generator(val_samples,
                                                            val_labels,
                                                            batch_size,
                                                            inception_preprocess_input,
                                                            augment=False,
                                                            image_size=(224, 224, 3))

    nbr_train_samples = len(train_samples)
    nbr_val_samples = len(val_samples)

    # call to fit using a generator
    history = inception_based_model.fit_generator(
        generator=training_sample_generator,
        steps_per_epoch=math.ceil(nbr_train_samples / batch_size),
        epochs=nbr_epochs,
        validation_data=validation_sample_generator,
        validation_steps=math.ceil(nbr_val_samples / batch_size),
        callbacks=cb,
        verbose=1)


def create_simpler_inception_based_model():
    """
    Inception-based model, uses InceptionV3 network without top layer and using max global pooling. Custom top layer
    is added to perform classification. This top layer is simpler and leaves a larger layer with more neurons before the
    softmax output layer which allows to classify according to more parameters and gives us better results.

    :return: the model.
    """

    # weights are pre-trained with imagenet
    base_model = InceptionV3(include_top=False, weights='imagenet', pooling='max', input_shape=(224, 224, 3))

    x = base_model.output
    x = Dense(2048, activation='relu', name='fc_1')(x)
    x = Dense(1024, activation='relu', name='fc_2')(x)
    predictions = Dense(nbr_classes, activation='softmax', name='fc_class')(x)  # dense layer with neurons with softmax
    model = Model(inputs=base_model.inputs, outputs=predictions)  # input is based model input, output is custom

    # by default only the fc layers are trainable (layers in base_model are not trainable)
    for layer in base_model.layers:
        layer.trainable = False

    return model


def train_simpler_inception_based_model(dataset_path,
                                        fine_tune_existing=None,
                                        save_path="best_trained_save.h5",
                                        freeze=True,
                                        learning_rate=0.001,
                                        percentage=0.9,
                                        nbr_epochs=10,
                                        batch_size=32):
    """
    Creates and train a simpler InceptionV3-based model on the fire images dataset or fine-tunes an pre-trained model
    with a custom learning rate. Some layers of the model can be frozen for training.

    :param dataset_path: path to the dataset.
    :param fine_tune_existing: whether a model was already trained and to just continue fine-tuning it. Its value should
    be the path to the existing model which needs to be loaded or None if no prior model is to be loaded.
    :param freeze: whether to freeze every layers except the fc at the end of the network.
    :param learning_rate: when fine-tuning, the learning rate can be specified.
    :param percentage: percentage of samples to be used for training. Must be in [0,1].
    :param nbr_epochs: number of epochs.
    :param batch_size: number of batches.
    """

    #  if a pre-trained model is specified, load it. Else create the model.
    if fine_tune_existing is not None:
        simpler_inception_based_model = load_model(fine_tune_existing)
    else:
        simpler_inception_based_model = create_simpler_inception_based_model()

    #  if no layer is to be frozen, set every layer to be trainable. Else, freeze every layer but layers fc_1, fc_2 and
    #  fc_class at the end of the network.
    if not freeze:
        for layer in simpler_inception_based_model.layers:
            layer.trainable = True
    else:
        for layer in simpler_inception_based_model.layers:
            if layer.name != 'fc_1' and layer.name != 'fc_2' and layer.name != 'fc_class':
                layer.trainable = False

    simpler_inception_based_model_save_folder = "model-saves/Inception_based/"

    # create save path
    if not os.path.exists(simpler_inception_based_model_save_folder):
        os.makedirs(simpler_inception_based_model_save_folder)

    simpler_inception_based_model_save_path = simpler_inception_based_model_save_folder + save_path

    simpler_inception_based_model.summary()

    # We can do learning rate adaptation later as part of fine tuning or use adaptive optimizer (rmsprop, adam)
    # keras.callbacks.callbacks.LearningRateScheduler(schedule, verbose=0)
    # keras.callbacks.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto',
    # min_delta=0.0001, cooldown=0, min_lr=0)
    # EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto',baseline=None, res
    # tore_best_weights=False)

    # saves the model when validation accuracy improves, overwrites previously saved model
    save_on_improve = ModelCheckpoint(simpler_inception_based_model_save_path,
                                      monitor='val_acc',
                                      verbose=1,
                                      save_best_only=True,
                                      save_weights_only=False,
                                      mode='max')

    # callbacks
    cb = [save_on_improve]

    # loss is categorical since we are classifying
    # if a pre-trained model was specified, we are fine tuning and need to take the custom learning rate into account
    if fine_tune_existing is not None:
        sgd = SGD(lr=learning_rate, momentum=0.0, nesterov=False)  # default lr is 0.01
        simpler_inception_based_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    else:
        simpler_inception_based_model.compile(loss='categorical_crossentropy', optimizer="sgd", metrics=['accuracy'])

    # extract the image paths to give to the generator
    (train_samples, train_labels), (val_samples, val_labels) = extract_dataset(dataset_path, classes, percentage)

    training_sample_generator = augmented_batch_generator(train_samples,
                                                          train_labels,
                                                          batch_size,
                                                          inception_preprocess_input,
                                                          augment=True,
                                                          image_size=(224, 224, 3))

    validation_sample_generator = augmented_batch_generator(val_samples,
                                                            val_labels,
                                                            batch_size,
                                                            inception_preprocess_input,
                                                            augment=False,
                                                            image_size=(224, 224, 3))

    nbr_train_samples = len(train_samples)
    nbr_val_samples = len(val_samples)

    # call to fit using a generator
    history = simpler_inception_based_model.fit_generator(
        generator=training_sample_generator,
        steps_per_epoch=math.ceil(nbr_train_samples / batch_size),
        epochs=nbr_epochs,
        validation_data=validation_sample_generator,
        validation_steps=math.ceil(nbr_val_samples / batch_size),
        callbacks=cb, verbose=1)
