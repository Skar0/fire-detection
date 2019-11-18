import math
import os

import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from matplotlib import pyplot as plt

from setup.naive_approach import extract_dataset, generate_from_paths_and_labels

"""
This module contains functions used to evaluate our models, both grahically on example images and on test sets.
A function also allows to extract examples classified with a low confidence by the network.
"""


def graphically_evaluate_model(model_path, classes_names, test_image_dir, preprocess_input, image_size=(224, 224)):
    """
    Loads a model, performs prediction on each image in test_image_dir and displays the image with the class name on
    top of it.

    :param model_path: path to the model.
    :param classes_names: names of the classes.
    :param test_image_dir: path to the test image directory.
    :param preprocess_input: preprocessing function for the network.
    :param image_size: size of the input image for the network.
    """

    nbr_classes = len(classes_names)

    model = load_model(model_path)

    # for each image
    for test_image_path in os.listdir(test_image_dir):

        # load image using keras
        img = image.load_img(test_image_dir + "/" + test_image_path, target_size=image_size)

        # processed image to feed the network
        processed_img = image.img_to_array(img)
        processed_img = np.expand_dims(processed_img, axis=0)
        processed_img = preprocess_input(processed_img)

        # get prediction using the network
        predictions = model.predict(processed_img)[0]

        # transform [0,1] values into percentages and associate it to its class name (class_name order was used to
        # one-hot encode the classes)
        result = [(classes_names[i], float(predictions[i]) * 100.0) for i in range(nbr_classes)]

        # sort the result by percentage
        result.sort(reverse=True, key=lambda x: x[1])

        # load image for displaying
        img = cv2.imread(test_image_dir + "/" + test_image_path)

        # transform into RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        font = cv2.FONT_HERSHEY_COMPLEX

        # write class percentages on the image
        for i in range(nbr_classes):

            # get the class and probability
            (class_name, prob) = result[i]

            textsize = cv2.getTextSize(class_name, font, 1, 2)[0]
            textX = (img.shape[1] - textsize[0]) / 2
            textY = (img.shape[0] + textsize[1]) / 2

            # print max probability prediction on top of the image
            if i == 0:
                cv2.putText(img, class_name, (int(textX) - 100, int(textY)), font, 5, (255, 255, 255), 6, cv2.LINE_AA)

            print("Class name: %s" % class_name)
            print("Probability: %.2f%%" % prob)

        plt.imshow(img)
        plt.show()


def evaluate_model(model_path, classes, preprocessing, dataset_path):
    """
    Loads a model and evaluates the model (metrics) on images provided in folder a dataset.

    :param model_path: path to the model.
    :param classes: names of the classes.
    :param preprocessing: preprocessing function for the network.
    :param dataset_path: path to the test dataset.
    """

    # for simplicity, the dataset is loaded using 99.9% of images
    (train_samples, train_labels), (val_samples, val_labels) = extract_dataset(dataset_path, classes, 0)

    batch_size = 16

    nbr_val_samples = len(val_samples)

    # create a generator from the naive approach (so we don't perform data augmentation on these test images)
    validation_sample_generator = generate_from_paths_and_labels(val_samples,
                                                                 val_labels,
                                                                 batch_size,
                                                                 preprocessing,
                                                                 image_size=(224, 224, 3))

    model = load_model(model_path)

    metrics = model.evaluate_generator(validation_sample_generator,
                                       steps=math.ceil(nbr_val_samples / 16),
                                       max_queue_size=10,
                                       workers=1,
                                       use_multiprocessing=True,
                                       verbose=1)

    # print the metrics
    out = ""
    for i in range(len(model.metrics_names)):
        out += model.metrics_names[i]
        out += " : "
        out += str(float(metrics[i]))
        out += " | "

    return out


def extract_hard_samples(model_path, preprocess_input, dataset_path, threshold, image_size=(224, 224)):
    """
    Extracts samples which are ard to classify for the network. Takes a dataset and a model as input, prediction is
    performed by the model on the samples from the dataset specified by dataset_path and samples with a classification
    confidence for the correct class lower than threshold are saved to a list.

    :param model_path: path to the model.
    :param preprocess_input: preprocessing function for the network.
    :param dataset_path: path to the dataset.
    :param threshold: confidence threshold for the prediction.
    :param image_size: size of the image fed to the network.
    """

    classes = ['fire', 'no_fire', 'start_fire']

    nbr_classes = 3

    model = load_model(model_path)

    # paths to hard examples are saved
    hard_examples = [[] for j in range(nbr_classes)]

    # for each class
    for i in range(nbr_classes):

        class_name = classes[i]

        # for each sample of that class
        for sample_path in os.listdir(dataset_path + class_name):

            img = image.load_img(dataset_path + class_name + "/" + sample_path, target_size=image_size)

            # processed image to feed the network
            processed_img = image.img_to_array(img)
            processed_img = np.expand_dims(processed_img, axis=0)
            processed_img = preprocess_input(processed_img)

            # get prediction using the network
            predictions = model.predict(processed_img)[0]

            # if prediction is not satisfactory
            if float(predictions[i]) < threshold:
                hard_examples[i].append(sample_path)

    return hard_examples


def display_hard_samples(hard_examples, dataset_path):
    """
    Displays samples that are hard to classify from the dataset specified by dataset_path, hard_examples should be a
    1x3 list containing paths of hard to classify samples.

    :param hard_examples: the list of paths to difficult examples.
    :param dataset_path: path to the dataset.
    """

    classes = ['fire', 'no_fire', 'start_fire']

    nbr_classes = 3

    for i in range(nbr_classes):

        class_name = classes[i]

        print("========== CLASS : " + class_name + " ==========")
        for sample_path in hard_examples[i]:

            # load image for displaying
            img = cv2.imread(dataset_path + "/" + class_name + "/" + sample_path)
            # transform into RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            plt.imshow(img)
            plt.show()
