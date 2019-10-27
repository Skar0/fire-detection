from keras import layers, backend, models
from keras import utils as keras_utils
import numpy as np

import os
import imghdr
import math

from keras.callbacks import ModelCheckpoint, TensorBoard
from matplotlib import pyplot as plt
from keras import Model
from keras.engine.saving import load_model

from keras.utils import np_utils
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications import imagenet_utils

import tensorflow as tf

CONS = 0
whole_printer = True
# global tf.name_scope

size_rep_dict = {'small': 0, 'medium': 3, 'big': 4}


def download_and_setup_dataset(size='small'):  # size in {'small', 'medium', 'big'}

    if size != 'full':
        # get number of repetitions
        rep = size_rep_dict[size]

        prefix = 'wget https://github.com/belarbi2733/keras_yolov3/releases/download/1/'
        inter_command = 'defi1certif-datasets-fire_' + size + '.tar'

        # get dataset
        if rep:
            for i in range(1, rep + 1):
                suffix = '.00' + str(i)

                command = prefix + inter_command + suffix
                print('+ executing: ' + command, flush=True)

                os.system(command)

            # recombine when needed
            recombine_command = 'cat'
            for i in range(1, rep + 1):
                suffix = '.00' + str(i)

                recombine_command += ' ' + inter_command + suffix

            recombine_command += ' > ' + inter_command

            print('+ executing: ' + recombine_command, flush=True)
            os.system(recombine_command)

        else:
            print('+ executing: ' + prefix + inter_command, flush=True)
            os.system(prefix + inter_command)

        # created dirs
        datasets_path = "datasets"
        print("- attempting to create 'datasets' directory", flush=True)
        if os.path.exists(datasets_path) == False:
            print("- creating 'datasets' directory", flush=True)
            os.makedirs(datasets_path)
        else:
            print("- 'datasets' directory already exists", flush=True)

        # put the each dataset in its asociate folder
        prefix = 'tar xf '
        suffix = " -C 'datasets' --one-top-level && mv datasets/defi1certif-datasets-fire_" + size + " datasets/" + size

        # execute
        command = prefix + inter_command + suffix
        print('+ executing: ' + command, flush=True)
        os.system(command)

        print('- ' + size + ' dataset successfully setup', flush=True)

    else:
        for key in size_rep_dict:
            download_and_setup_dataset(size=key)


def setup_full_dataset():
    """
    Combines all datasets in a single folder.
    :return:
    """

    print('- fusioning all datasets', flush=True)

    print('- creating directories', flush=True)
    # creating the folder to merge datasets
    if not os.path.exists("datasets/all"):
        os.makedirs("datasets/all")
    if not os.path.exists("datasets/all/fire"):
        os.makedirs("datasets/all/fire")
    if not os.path.exists("datasets/all/no_fire"):
        os.makedirs("datasets/all/no_fire")
    if not os.path.exists("datasets/all/start_fire"):
        os.makedirs("datasets/all/start_fire")

    print('- moving files', flush=True)
    # moving images from the small dataset to the full dataset
    print('+ executing: ' + "find datasets/small/fire -type f -print0 | xargs -0 mv -t datasets/all/fire/", flush=True)
    os.system("find datasets/small/fire -type f -print0 | xargs -0 mv -t datasets/all/fire/")

    print('+ executing: ' + "find datasets/small/no_fire -type f -print0 | xargs -0 mv -t datasets/all/no_fire/",
          flush=True)
    os.system("find datasets/small/no_fire -type f -print0 | xargs -0 mv -t datasets/all/no_fire/")

    print('+ executing: ' + "find datasets/small/start_fire -type f -print0 | xargs -0 mv -t datasets/all/start_fire/",
          flush=True)
    os.system("find datasets/small/start_fire -type f -print0 | xargs -0 mv -t datasets/all/start_fire/")

    # moving images from the medium dataset to the full dataset
    print('+ executing: ' + "find datasets/medium/fire -type f -print0 | xargs -0 mv -t datasets/all/fire/", flush=True)
    os.system("find datasets/medium/fire -type f -print0 | xargs -0 mv -t datasets/all/fire/")

    print('+ executing: ' + "find datasets/medium/no_fire -type f -print0 | xargs -0 mv -t datasets/all/no_fire/",
          flush=True)
    os.system("find datasets/medium/no_fire -type f -print0 | xargs -0 mv -t datasets/all/no_fire/")

    print('+ executing: ' + "find datasets/medium/start_fire -type f -print0 | xargs -0 mv -t datasets/all/start_fire/",
          flush=True)
    os.system("find datasets/medium/start_fire -type f -print0 | xargs -0 mv -t datasets/all/start_fire/")

    # moving images from the large dataset to the full dataset
    print('+ executing: ' + "find datasets/big/fire -type f -print0 | xargs -0 mv -t datasets/all/fire/", flush=True)
    os.system("find datasets/big/fire -type f -print0 | xargs -0 mv -t datasets/all/fire/")

    print('+ executing: ' + "find datasets/big/no_fire -type f -print0 | xargs -0 mv -t datasets/all/no_fire/",
          flush=True)
    os.system("find datasets/big/no_fire -type f -print0 | xargs -0 mv -t datasets/all/no_fire/")

    print('+ executing: ' + "find datasets/big/start_fire -type f -print0 | xargs -0 mv -t datasets/all/start_fire/",
          flush=True)
    os.system("find datasets/big/start_fire -type f -print0 | xargs -0 mv -t datasets/all/start_fire/")

    print("- files moved to:'datasets/all/", flush=True)

    space = len("- files moved to:'datasets/all/")
    print(' ' * space + "fire'", flush=True)
    print(' ' * space + "no_fire'", flush=True)
    print(' ' * space + "start_fire'", flush=True)

    print('- done', flush=True)


def download_and_setup_full_dataset():
    download_and_setup_dataset('full')
    setup_full_dataset()


def generate_from_paths_and_labels(images_paths, labels, batch_size, image_size=(224, 224),
                                   preprocessing=preprocess_input):
    """
    Generator to give to the fit function, generates batches of samples for training.
    This avoids to load the full dataset in memory. This can also be a Keras class.
    :param images_paths:
    :param labels:
    :param batch_size:
    :param image_size:
    :param preprocessing:
    :return:
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

            # preprocessing might notably normalize between 0 and 1 the RGB values
            inputs = preprocess_input(inputs)

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

    # ignore hidden files
    def listdir_nohidden(path):
        for f in os.listdir(path):
            if not f.startswith('.'):
                yield f

    # putting images paths and labels in lists
    images_paths, labels = [], []
    for class_name in listdir_nohidden(dataset_path):
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


# this will be a virgin network, no preloaded weights, we will train it from
# scratch


# wrapper for a convolution layer
def convolution2d_layer(tensor, filters, kernel_shape, name, padding='same', strides=(1, 1)):
    tensor = layers.Conv2D(filters, kernel_shape, strides=strides, padding=padding, name=name)(tensor)
    return tensor


# wrapper for the batch normalization layer
def batch_norma_layer(tensor, name):
    """
    this paper justifies the gain in speed during training by reducing the
    'internal covariance shift' and thus accelerating the convergence:
    https://arxiv.org/pdf/1502.03167.pdf

    :return: a normalized layer
    """

    # this depends only of the backend used 'theano', 'tensorflow',...
    if backend.image_data_format() == 'channels_first':
        axis = 1
    else:
        axis = 3

    tensor = layers.BatchNormalization(axis=axis, scale=False, name=name)(tensor)

    return tensor


# this function encapsulates both procedures: convolution and batch normalisation
def cvbn(tensor, filters, kernel_shape, name='custom', padding='same', strides=(1, 1)):
    global CONS
    CONS += 1
    with tf.name_scope('CustomOP') as scope:
        tensor = convolution2d_layer(tensor, filters=filters,
                                     kernel_shape=kernel_shape,
                                     name=name + '_conv' + str(CONS), padding=padding, strides=strides)
        tensor = batch_norma_layer(tensor, name=name + '_bn' + str(CONS))
        tensor = layers.Activation('relu', name=name + str(CONS))(tensor)

    return tensor


# this is a linear meta-layer for preprocessing the input
def type0_layer(tensor, name,
                pooling_rep=2,
                body_by_rep=None,
                type_by_rep=None,
                filter_list=None,
                kernel_shape_list=None,
                padding_list=None,
                pooling_kernel_shape_list=None,
                pooling_strides_list=None):
    """
    linear layer, stem for preprocess the input before the inception-like layers

    :param tensor: a tensor to be modified
    # :param name: name of the result
    # :param axis: depends only of the backend used, should be 1 if 'channel_first'
    :param pooling_rep: number of times a pooling will be executed during the treatment, minimum 1
    :param body_by_rep: number of convolution layers before each pooling layer
    :param type_by_rep: type of pooling ('max', 'avg'), length should be equal to pooling_rep
    :param filter_list: filters for the inner convolution layers, length should be equal to sum of values
                        in 'body_by_rep'
    :param kernel_shape_list: same as filter_list
    :param padding_list: same as filter_list
    :param pooling_kernel_shape_list: shape of kernel for the pooling layers, length should be 'pooling_rep'
    :param pooling_strides_list: same as pooling_kernel_shape
    :return: a modified tensor
    """

    if body_by_rep is None:
        body_by_rep = [3, 2]
    if type_by_rep is None:
        type_by_rep = ['max', 'max']
    if filter_list is None:
        filter_list = [32, 32, 64, 80, 192]
    if kernel_shape_list is None:
        kernel_shape_list = [(3, 3), (3, 3), (3, 3), (1, 1), (3, 3)]
    if padding_list is None:
        padding_list = ['valid' for i in range(5)]
        padding_list[2] = 'same'
    if pooling_kernel_shape_list is None:
        pooling_kernel_shape_list = [(3, 3), (3, 3)]
    if pooling_strides_list is None:
        pooling_strides_list = [(2, 2) for i in range(2)]

    counter = 0

    for i in range(pooling_rep):
        size_of_inner_layer = body_by_rep[i]

        for j in range(size_of_inner_layer):
            tensor = cvbn(tensor, filters=filter_list[counter], kernel_shape=kernel_shape_list[counter],
                          padding=padding_list[counter])
            counter += 1

        if type_by_rep[i] == 'max':
            tensor = layers.MaxPooling2D(pooling_kernel_shape_list[i], pooling_strides_list[i], name=name + '_' +
                                                                                                     str(i))(tensor)
        else:
            tensor = layers.AvgPooling2D(pooling_kernel_shape_list[i], pooling_strides_list[i], name=name + '_' +
                                                                                                     str(i))(tensor)

    return tensor


def type1_layer(tensor, name, axis,
                width=4,
                inner_pooling='avg',
                rep_by_branch=None,
                filter_list=None,
                kernel_shape_list=None,
                strides_list=None,
                padding_list=None,
                pooling_time=True,
                pooling_filter=32, pooling_kernel_shape=(1, 1),
                pooling_padding='same',
                use_cvbn_pooling=True,
                pooling_strides=(1, 1)
                ):
    """
    inception layer, using the InceptionV3 structure, this meta-layer is totally modifiable, first recurrent structure

    :param tensor: a tensor to be modified
    :param name:  name of the result
    :param axis: axis for the batch normalization layer
    :param width: width of the network, counting the pooling branch, the user cannot remove this branch
    :param inner_pooling: type of pooling ('avg', 'max')
    :param rep_by_branch: a list with ('width'-1) length, the last branch is always the pooling branch
    :param filter_list: a list of filters to be used, the length should be equal to the sum of elements in
                        'rep_by_branch'
    :param kernel_shape_list: same as filter_list
    :param strides_list: sames as filter_list
    :param padding_list: same as filter_list
    :param pooling_time: if set to 'True' a supplementary branch will be used for pooling
    :param pooling_filter: filter for the pooling layer
    :param pooling_kernel_shape: kernel_shape for the pooling layer
    :param pooling_padding: padding to be used if pooling is set to 'True'
    :param use_cvbn_pooling: if set to 'True' will pass a cvbn before concatenation
    :param pooling_strides: strides for the pooling phase
    :return:
    """

    if rep_by_branch is None:
        rep_by_branch = [1, 2, 3]
    if filter_list is None:
        filter_list = [64, 48, 64, 64, 96, 96]
    if kernel_shape_list is None:
        kernel_shape_list = [(1, 1), (1, 1), (5, 5), (1, 1), (3, 3), (3, 3)]
    if strides_list is None:
        strides_list = [(1, 1) for i in range(6)]
    if padding_list is None:
        padding_list = ['same' for i in range(6)]

    # keep count of overall passes and layers to be concatenated at the end
    layers_to_concatenate = []
    counter = 0

    # each pass on the loop is a branch
    for i in range(width - 1):
        # number of times this branch is executed
        repetitions = rep_by_branch[i]

        # to know if is the first time we enter in this branch
        first_time = True

        for rep in range(repetitions):
            if first_time:
                branch = cvbn(tensor, filter_list[counter], kernel_shape=kernel_shape_list[counter],
                              strides=strides_list[counter], padding=padding_list[counter])

                first_time = False
            else:
                branch = cvbn(branch, filter_list[counter], kernel_shape=kernel_shape_list[counter],
                              strides=strides_list[counter], padding=padding_list[counter])

            # the counter is global for all layers
            counter += 1

        # reset value
        first_time = True

        # branch end, we added the value to concatenate
        layers_to_concatenate.append(branch)

    # pooling time
    if pooling_time:
        if inner_pooling == 'avg':
            tensor_pooling = layers.AveragePooling2D((3, 3), strides=pooling_strides, padding=pooling_padding)(tensor)
        else:
            tensor_pooling = layers.MaxPooling2D((3, 3), strides=pooling_strides, padding=pooling_padding)(tensor)
        if use_cvbn_pooling:
            tensor_pooling = cvbn(tensor_pooling, pooling_filter, kernel_shape=pooling_kernel_shape)

        # add 'tensor_pooling' to concatenation values
        layers_to_concatenate.append(tensor_pooling)

    # concatenate and return
    tensor = layers.concatenate(layers_to_concatenate, axis=axis, name=name)
    return tensor


"""
We are going to mimic the structure of InceptionV3, but with some changes.
With only one phenomena to train, we do not need a so complex network.
Maybe a more shallow network. But the global structure is to be preserved.
First a stem meta-layer,
Second a couple of inception and reduction layers,
Third and last an output layer. 
"""


def Cladoh(include_top=True, input_tensor=None, input_shape=None, pooling=None, classes=3):
    # include_top to be let as it is
    # weights to None for virgin network
    # input_tensor to None
    # input_shape I think is going to take (244,244,3) as default value
    # pooling need more information
    # classes is going to be 3 by default
    # with tf.name_scope('Pierre') as scope:
    # maybe this need to be changed
    if input_shape is None:
        input_shape = (224, 224, 3)

    # input preparation
    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # axis definition, it depends of the backend
    if backend.image_data_format() == 'channels_first':
        axis = 1
    else:
        axis = 3

    # from here is the layers definition

    # stem layer
    # print('+'*40, 'img_input_shape: ', img_input.shape)
    tensor = type0_layer(img_input, name='stem')

    # pair of inception and reduction layers:

    # inception layer
    # three passes
    tensor = type1_layer(tensor, 'inception1', axis)
    tensor = type1_layer(tensor, 'inception2', axis)
    tensor = type1_layer(tensor, 'inception3', axis)

    # reduction layer
    reduction_layer_width = 3
    reduction_layer_rep_by_branch = [1, 3]
    reduction_layer_filter_list = [384, 64, 96, 96]
    reduction_layer_kernel_shape_list = [(3, 3), (1, 1), (3, 3), (3, 3)]
    reduction_layer_strides_list = [(2, 2), (1, 1), (1, 1), (2, 2)]
    reduction_layer_padding_list = ['valid', 'same', 'same', 'valid']

    tensor = type1_layer(tensor, 'reduction1', axis,
                         width=reduction_layer_width,
                         inner_pooling='max',
                         rep_by_branch=reduction_layer_rep_by_branch,
                         filter_list=reduction_layer_filter_list,
                         kernel_shape_list=reduction_layer_kernel_shape_list,
                         strides_list=reduction_layer_strides_list,
                         pooling_time=True,
                         pooling_padding='valid',
                         padding_list=reduction_layer_padding_list,
                         use_cvbn_pooling=False,
                         pooling_strides=(2, 2))

    # TODO: maybe the change of dimension is to abrupt, something with a more
    #     : nice slope ?

    # now last layers, there is an augmentation of dimension from now to
    # the output layer

    for i in range(2):
        """
        augmentation_layer_width = 3
        augmentation_layer_rep_by_branch = []
        augmentation_layer_filter_list = [384, 64, 96, 96]
        augmentation_layer_kernel_shape_list = [(3, 3), (1, 1), (3, 3), (3, 3)]
        augmentation_layer_strides_list = [(2, 2), (1, 1), (1, 1), (2, 2)]
        augmentation_layer_padding_list = ['valid', 'same', 'same', 'valid']
        """

        # TODO: you need to define a type2 layer for this structure
        tensor_1x1 = cvbn(tensor, filters=320, kernel_shape=(1, 1))

        tensor_3x3 = cvbn(tensor, filters=384, kernel_shape=(1, 1))
        tensor_3x3_1 = cvbn(tensor_3x3, filters=384, kernel_shape=(1, 3))
        tensor_3x3_2 = cvbn(tensor_3x3, filters=384, kernel_shape=(3, 1))
        tensor_3x3 = layers.concatenate(
            [tensor_3x3_1, tensor_3x3_2],
            axis=axis,
            name='augmentation' + str(i))

        tensor_3x3dbl = cvbn(tensor, filters=448, kernel_shape=(1, 1))
        tensor_3x3dbl = cvbn(tensor_3x3dbl, filters=384, kernel_shape=(3, 3))
        tensor_3x3dbl_1 = cvbn(tensor_3x3dbl, filters=384, kernel_shape=(1, 3))
        tensor_3x3dbl_2 = cvbn(tensor_3x3dbl, filters=384, kernel_shape=(3, 1))

        tensor_3x3dbl = layers.concatenate(
            [tensor_3x3dbl_1, tensor_3x3dbl_2], axis=axis)

        tensor_pooling = layers.AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(tensor)
        tensor_pooling = cvbn(tensor_pooling, filters=192, kernel_shape=(1, 1))
        tensor = layers.concatenate(
            [tensor_1x1, tensor_3x3, tensor_3x3dbl, tensor_pooling],
            axis=axis,
            name='augmentation2' + str(9 + i))
    # from here is the final setup and return

    # output layers
    if include_top:
        # Classification block
        tensor = layers.GlobalAveragePooling2D(name='avg_pool')(tensor)
        tensor = layers.Dense(classes, activation='softmax', name='predictions')(tensor)
    else:
        if pooling == 'avg':
            tensor = layers.GlobalAveragePooling2D()(tensor)
        elif pooling == 'max':
            tensor = layers.GlobalMaxPooling2D()(tensor)

    # print('+'*40, 'tensor_shape: ', tensor.shape)
    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # model creation and return
    model = models.Model(inputs, tensor, name='cladoh')
    return model


# we use the same preprocessing as in inception
def preprocess_input_custom(x, **kwargs):
    """Preprocesses a numpy array encoding a batch of images.
    # Arguments
        x: a 4D numpy array consists of RGB values within [0, 255].
    # Returns
        Preprocessed array.
    """
    return imagenet_utils.preprocess_input(x, mode='tf', **kwargs)


def train_and_save_custom_model(dataset_path, percentage=0.8, nbr_epochs=10, batch_size=32):
    """
    :param percentage: percentage of samples to be used for training. Must be in [0,1].
    :param nbr_epochs:
    :param batch_size:
    """

    model = Cladoh(include_top=True, pooling='max', input_shape=(224, 224, 3))

    Custom_based_model_save_folder = "model-saves/Custom_based/"

    # create save path
    if not os.path.exists(Custom_based_model_save_folder):
        os.makedirs(Custom_based_model_save_folder)

    Custom_based_model_save_path = Custom_based_model_save_folder + "best_trained_save.h5"

    # checkpoints

    # We can do learning rate adaptation later as part of fine tuning or use adaptive optimizer (rmsprop, adam)
    # keras.callbacks.callbacks.LearningRateScheduler(schedule, verbose=0)
    # keras.callbacks.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto',
    # min_delta=0.0001, cooldown=0, min_lr=0)

    # saves the model when validation accuracy improves
    save_on_improve = ModelCheckpoint(Custom_based_model_save_path, monitor='val_accuracy', verbose=1,
                                      save_best_only=True, save_weights_only=False, mode='max')

    # EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto',baseline=None, res
    # tore_best_weights=False)

    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True,
                              write_grads=False, write_images=False, embeddings_freq=0,
                              embeddings_layer_names=None, embeddings_metadata=None,
                              embeddings_data=None, update_freq='epoch')

    callbacks = [save_on_improve, tensorboard]

    # loss is categorical since we are classifying
    model.compile(loss='categorical_crossentropy', optimizer="sgd", metrics=['accuracy'], )
    # callbacks=callbacks)

    (train_samples, train_labels), (val_samples, val_labels) = extract_dataset(dataset_path, classes, percentage)

    if whole_printer:
        print(train_samples.shape)
        print(train_labels.shape)
        print(val_samples.shape)
        print(val_labels.shape)
    training_sample_generator = generate_from_paths_and_labels(train_samples, train_labels, batch_size,
                                                               image_size=(224, 224, 3))

    validation_sample_generator = generate_from_paths_and_labels(val_samples, val_labels, batch_size,
                                                                 image_size=(224, 224, 3))

    nbr_train_samples = len(train_samples)
    nbr_val_samples = len(val_samples)

    # call to fit using a generator
    history = model.fit_generator(
        generator=training_sample_generator,
        steps_per_epoch=math.ceil(nbr_train_samples / batch_size),
        epochs=nbr_epochs,
        validation_data=validation_sample_generator,
        validation_steps=math.ceil(nbr_val_samples / batch_size),
        verbose=1)

    model.save(Custom_based_model_save_path)

    print('model saved to: ', Custom_based_model_save_path)
    return Custom_based_model_save_path, history


if __name__ == '__main__':
    # some initial values

    classes = ['fire', 'no_fire', 'start_fire']
    nbr_classes = 3

    classes_value = classes
    split_percentage = 0.8  # @param {type:"slider", min:0.3, max:0.9, step:0.1}

    nbr_batch_size = 64  # @param [1,2,4,8,16,32,64,128, 256] {type:"raw"}
    dataset_name = 'big'  # @param ["small","medium","big","all"]

    dataset_path = os.path.join('datasets/', dataset_name)
    epochs = 30  # @param {type:"slider", min:5, max:100, step:5}

    print('nbr_classes: ', nbr_classes)
    print('classes_value: ', classes_value)
    print('split_percentage: ', split_percentage)
    print('nbr_batch_size: ', nbr_batch_size)
    print('dataset_name: ', dataset_name)
    print('dataset_path: ', dataset_path)
    print('epochs: ', epochs)

    # download and setup dataset
    if not os.path.exists(dataset_path):
        download_and_setup_dataset(dataset_name)

    # remove anything problematic
    os.system('rm -r ' + dataset_path + '/de*')

    # define, train and save Custom model
    # this function returns the path where the model has been saved to

    model_path, history = train_and_save_custom_model(dataset_path, percentage=split_percentage, nbr_epochs=epochs,
                                                      batch_size=nbr_batch_size)

