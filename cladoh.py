from keras import layers, backend, models
from keras import utils as keras_utils

from keras.applications import imagenet_utils

import tensorflow as tf

CONS = 0
whole_printer = True


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

