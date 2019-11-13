import imghdr
import math
import os
import numpy as np
from keras import Model
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as inception_preprocess_input
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import GlobalAveragePooling2D, Dense
from keras.preprocessing import image
from keras.utils import np_utils

"""
This module contains naive transfer learning from VGG16 and InceptionV3 for fire detection.
"""

classes = ['fire', 'no_fire', 'start_fire']
nbr_classes = 3


def generate_from_paths_and_labels(images_paths, labels, batch_size, preprocessing, image_size=(224, 224)):
    """
    Generator to give to the fit function, generates batches of samples for training.
    This avoids to load the full dataset in memory. This can also be a Keras class.
    :param images_paths:
    :param labels:
    :param batch_size:
    :param image_size:
    :param preprocessing:
    """

    number_samples = len(images_paths)

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
            # converting the loaded images to numpy arrays
            inputs = np.array(list(map(
                lambda x: image.img_to_array(x),
                inputs
            )))

            # preprocessing the batch might notably normalize between 0 and 1 the RGB values, this is model-dependant
            inputs = preprocessing(inputs)

            # yields the image batch and corresponding labels
            yield (inputs, labels[i:i + batch_size])


def extract_dataset(dataset_path, classes_names, percentage):
    """
    Assumes that dataset_path/classes_names[0] is a folder containing all images of class classes_names[0].
    All image paths are loaded into a numpy array, corresponding labels are one-hot encoded and put into a numpy array.
    Samples are shuffled before splitting into training and validation sets to prevent problems since samples are loaded
    in order of their class.

    :param dataset_path: path to the root of the dataset.
    :param classes_names: names of the classes.
    :param percentage: percentage of samples to be used for training, the rest is for validation. Must be in [0,1].
    :return: (x_train, y_train), (x_val, y_val) a list of image paths and a list of corresponding labels for training
    and validation.
    """

    num_classes = len(classes_names)

    # putting images paths and labels in lists
    images_paths, labels = [], []
    for class_name in os.listdir(dataset_path):
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

    # one-hot encode the labels
    labels_oh = np_utils.to_categorical(labels, num_classes)
    # convert images_paths to numpy array to apply permutation
    images_paths = np.array(images_paths)

    number_samples = len(images_paths)
    perm = np.random.permutation(number_samples)
    labels_oh = labels_oh[perm]
    images_paths = images_paths[perm]

    # 90% of samples used for training
    border = math.floor(percentage * len(images_paths))

    train_labels, val_labels = labels_oh[:border], labels_oh[border:]
    train_samples, val_samples = images_paths[:border], images_paths[border:]

    print("Training on %d samples" % (len(train_samples)))
    print("Validation on %d samples" % (len(val_samples)))

    return (train_samples, train_labels), (val_samples, val_labels)


def create_VGG16_based_model():
    """
    VGG16-based model.
    :return: the model.
    """

    # the 3 fully connected layers at the end are not included
    # weights are pre-trained with imagenet
    base_model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)  # to research
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(nbr_classes, activation='softmax')(x)  # dense layer with 3 neurons with softmax
    model = Model(inputs=base_model.inputs, outputs=predictions)  # input is based model input, output is custom

    # we let every layer to be trainable
    for layer in model.layers:
        layer.trainable = True

    return model


def train_and_save_VGG16_based_model(dataset_path, percentage=0.9, nbr_epochs=10, batch_size=32):
    """
    :param percentage: percentage of samples to be used for training. Must be in [0,1].
    :param nbr_epochs:
    :param batch_size:
    """

    VGG16_based_model = create_VGG16_based_model()

    # loss is categorical since we are classifying
    VGG16_based_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    VGG16_based_model_save_folder = "model-saves/VGG16_based/"

    # create save path
    if not os.path.exists(VGG16_based_model_save_folder):
        os.makedirs(VGG16_based_model_save_folder)

    (train_samples, train_labels), (val_samples, val_labels) = extract_dataset(dataset_path, classes, percentage)

    training_sample_generator = generate_from_paths_and_labels(train_samples, train_labels, batch_size,
                                                               vgg16_preprocess_input, image_size=(224, 224, 3))

    validation_sample_generator = generate_from_paths_and_labels(val_samples, val_labels, batch_size,
                                                                 vgg16_preprocess_input, image_size=(224, 224, 3))

    nbr_train_samples = len(train_samples)
    nbr_val_samples = len(val_samples)

    # call to fit using a generator
    history = VGG16_based_model.fit_generator(
        generator=training_sample_generator,
        steps_per_epoch=math.ceil(nbr_train_samples / batch_size),
        epochs=nbr_epochs,
        validation_data=validation_sample_generator,
        validation_steps=math.ceil(nbr_val_samples / batch_size),
        verbose=1)

    VGG16_based_model_save_path = VGG16_based_model_save_folder + "trained_save.h5"
    VGG16_based_model.save(VGG16_based_model_save_path)


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


    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True,
                              write_grads=False, write_images=False, embeddings_freq=0,
                              embeddings_layer_names=None, embeddings_metadata=None,
                              embeddings_data=None, update_freq='epoch')

    cb = [save_on_improve, tensorboard]

    # loss is categorical since we are classifying
    Inception_based_model.compile(loss='categorical_crossentropy', optimizer="sgd", metrics=['accuracy'])

    (train_samples, train_labels), (val_samples, val_labels) = extract_dataset(dataset_path, classes, percentage)

    training_sample_generator = generate_from_paths_and_labels(train_samples, train_labels, batch_size,
                                                               inception_preprocess_input, image_size=(224, 224, 3))

    validation_sample_generator = generate_from_paths_and_labels(val_samples, val_labels, batch_size,
                                                                 inception_preprocess_input, image_size=(224, 224, 3))

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
