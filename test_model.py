import cv2
import numpy as np
import os
import math
from keras.engine.saving import load_model
from keras.preprocessing import image
from keras_applications.vgg16 import preprocess_input
from simple_models import extract_dataset, generate_from_paths_and_labels
from matplotlib import pyplot as plt


def graphically_test_model(model_path, classes_names, test_image_dir, image_size=(224, 224)):
    """
    Loads a model, does a prediction on each image in test_image_dir and displays the image with the class name on
    top of it.
    :param model_path:
    :param classes_names:
    :param test_image_dir:
    :param image_size:
    """
    nbr_classes = len(classes_names)
    model = load_model(model_path)

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
            (class_name, prob) = result[i]
            textsize = cv2.getTextSize(class_name, font, 1, 2)[0]
            textX = (img.shape[1] - textsize[0]) / 2
            textY = (img.shape[0] + textsize[1]) / 2
            if (i == 0):
                cv2.putText(img, class_name, (int(textX) - 100, int(textY)), font, 5, (255, 255, 255), 6, cv2.LINE_AA)
            print("Top %d ====================" % (i + 1))
            print("Class name: %s" % (class_name))
            print("Probability: %.2f%%" % (prob))
        plt.imshow(img)
        plt.show()


def evaluate_model(model_path, classes, dataset_path):
    """
    Loads a model and evaluates the model (metrics) on images provided in folder a dataset.
    :param model_path:
    :param classes:
    :param test_dataset:
    """
    # For simplicity, the dataset is loaded using 99.9% of images
    (train_samples, train_labels), (val_samples, val_labels) = extract_dataset(dataset_path, classes, 0.001)
    batch_size = 16
    nbr_val_samples = len(val_samples)
    validation_sample_generator = generate_from_paths_and_labels(val_samples, val_labels, batch_size,
                                                                 image_size=(224, 224, 3))

    model = load_model(model_path)
    model.evaluate_generator(validation_sample_generator, steps=math.ceil(nbr_val_samples / 16), callbacks=None,
                             max_queue_size=10, workers=1, use_multiprocessing=True, verbose=1)
