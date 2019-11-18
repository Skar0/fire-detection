import os
import cv2
import imageio
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

"""
This module contains functions which process an mp4 video and annotate its frames with a prediction by a CNN on whether 
a fire is preset or not in the frame. The output is an annotated video.
"""


def video_fire_detection(input_video_path, output_video_path, model_path, model_preprocess, image_size, detection_freq):
    """
    Loads a video given by input_video_path, performs fire detection using the model saved in model_path then annotates
    frames of the video with the detected class and create an annotated video in output_video_path. For speed, not
    every frame is fed to the network for detection. One out of detection_freq frames is fed to the network for
    prediction and its prediction is used to annotate the subsequent frames until a new prediction is made. This is also
    sound given the 'static' nature of fire and its slow evolution, making subsequent frames somewhat similar. This
    version is much faster as frames are not written to the disk and are processed on the fly.

    :param input_video_path: input video (must be mp4).
    :param output_video_path: output video path.
    :param model_path: path to the neural network model.
    :param model_preprocess: preprocessing function for the model.
    :param image_size: size of the image, extracted from the video and fed to the network.
    :param detection_freq: prediction is done every detection_freq frames.
    """

    # images extracted from the video are saved to a directory
    if not os.path.exists("temp_frames"):
        os.makedirs("temp_frames")

    classes = ['fire', 'no_fire', 'start_fire']

    nbr_classes = 3

    video_writer = imageio.get_writer(output_video_path, fps=24)

    model = load_model(model_path)

    # loading the video
    video = cv2.VideoCapture(input_video_path)

    # opening the video
    if not video.isOpened():
        print("Error opening video stream or file")

    # frame numbering for the images
    frame_nbr = 0
    img, max_class, max_proba = None, "unknown", 0

    while video.isOpened():

        # capture a frame
        not_done, frame = video.read()

        # we are not finished reading
        if not_done:

            # save single frame to a temp folder
            img_name = "temp-frame" + str(frame_nbr) + ".png"
            img_path = "temp_frames/" + img_name

            cv2.imwrite(img_path, frame)

            # we do not perform a new prediction and use the previous one
            if frame_nbr % detection_freq != 0:

                # load image for writing the video
                img = cv2.imread(img_path)
                height, width, channels = img.shape

                # convert image to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # setup text
                font = cv2.FONT_HERSHEY_SIMPLEX
                text = str(max_class) + " : " + str("{:.2f}".format(max_proba)) + "%"

                # get boundary of this text
                textsize = cv2.getTextSize(text, font, 1, 2)[0]

                # get coordinates based on boundary
                textX = (img.shape[1] - textsize[0]) // 2
                textY = (img.shape[0] + textsize[1]) // 2

                # set the rectangle background to black
                rectangle_bgr = (0, 0, 0)

                # make the coordinates of the box and draw a box
                box_coords = ((textX, textY), (textX + textsize[0], textY - textsize[1]))
                cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)

                # add text centered on image
                cv2.putText(img, text, (textX, textY), font, 1, (255, 255, 255), 2)

            else:
                # prediction is performed, first we load the image
                img = image.load_img(img_path, target_size=image_size)
                img = image.img_to_array(img)
                img = np.expand_dims(img, axis=0)
                img = model_preprocess(img)

                # perform the prediction
                probabilities = model.predict(img,
                                              batch_size=1,
                                              verbose=0)[0]

                # transform [0,1] values into percentages and associate it to its class name
                result = [(classes[i], float(probabilities[i]) * 100.0) for i in range(nbr_classes)]

                # sort the result by percentage
                result.sort(reverse=True, key=lambda x: x[1])

                # take the class with max percentage
                max_class, max_proba = result[0][0], result[0][1]

                # load image for writing the video
                img = cv2.imread(img_path)

                # convert image to RGB (using cv2 this is required, not when using keras' function)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # setup text
                font = cv2.FONT_HERSHEY_SIMPLEX

                # probability is formatted to have two digits after the coma
                text = str(max_class) + " : " + str("{:.2f}".format(max_proba)) + "%"

                # get boundary of this text
                textsize = cv2.getTextSize(text, font, 1, 2)[0]

                # get coordinates based on boundary
                textX = (img.shape[1] - textsize[0]) // 2
                textY = (img.shape[0] + textsize[1]) // 2

                # set the rectangle background to black
                rectangle_bgr = (0, 0, 0)

                # make the coordinates of the box
                box_coords = ((textX, textY), (textX + textsize[0], textY - textsize[1]))
                cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)

                # add text centered on image
                cv2.putText(img, text, (textX, textY), font, 1, (255, 255, 255), 2)

            frame_nbr = frame_nbr + 1
            video_writer.append_data(img)

        else:
            break

    video_writer.close()
    video.release()


def extract_images_from_video(video_path, images_directory):
    """
    Extract frames from a video specified by video_path and writes them to the folder images_directory.

    :param video_path: the path to the mp4 video.
    :param images_directory: directory in which to write the images.
    """

    # images extracted from the video are saved to a directory
    if not os.path.exists(images_directory):
        os.makedirs(images_directory)

    # loading the video
    video = cv2.VideoCapture(video_path)

    # opening the video
    if not video.isOpened():
        print("Error opening video stream or file")

    # frame numbering for the images
    frame_nbr = 0

    while video.isOpened():

        # capture a frame
        not_done, frame = video.read()

        # we are not finished reading
        if not_done:
            # name the frame and save it as a png file
            img_name = "frame_" + str(frame_nbr) + ".png"
            img_path = images_directory + img_name
            cv2.imwrite(img_path, frame)
            frame_nbr = frame_nbr + 1
        else:
            break

    # free video
    video.release()


def detect_fire_save_frames(input_video_path, output_video_path, model_path, model_preprocess, image_size,
                            detection_freq):
    """
    Loads a video given by input_video_path, performs fire detection using the model saved in model_path then annotates
    frames of the video with the detected class and create an annotated video in output_video_path. For speed, not
    every frame is fed to the network for detection. One out of detection_freq frames is fed to the network for
    prediction and its prediction is used to annotate the subsequent frames until a new prediction is made. This is also
    sound given the 'static' nature of fire and its slow evolution, making subsequent frames somewhat similar. This
    version writes all frames to a directory names video_frames/ before predicting.

    :param input_video_path: input video (must be mp4).
    :param output_video_path: output video path.
    :param model_path: path to the neural network model.
    :param model_preprocess: preprocessing function for the model.
    :param image_size: size of the image, extracted from the video and fed to the network.
    :param detection_freq: prediction is done every detection_freq frames.
    """
    classes = ['fire', 'no_fire', 'start_fire']

    nbr_classes = 3

    extract_images_from_video(input_video_path, "./video_frames/")

    video_writer = imageio.get_writer(output_video_path, fps=24)

    model = load_model(model_path)

    max_class, max_proba = "unknown", 0

    # sort frames and apply detection every detection_freq frames
    frames = []
    counter = 0

    for img_path in sorted(os.listdir('video_frames'), key=lambda f: int("".join(list(filter(str.isdigit, f))))):

        complete_path = 'video_frames/' + img_path

        frames.append(complete_path)

        # load image for writing the video
        img = cv2.imread(complete_path)

        # convert image to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # setup text
        font = cv2.FONT_HERSHEY_SIMPLEX

        text = str(max_class) + " : " + str(max_proba) + "%"

        # get boundary of this text
        textsize = cv2.getTextSize(text, font, 1, 2)[0]

        # get coords based on boundary
        textX = (img.shape[1] - textsize[0]) // 2
        textY = (img.shape[0] + textsize[1]) // 2

        # print(img, text, (textX, textY), font, 1, (255, 255, 255), 2)

        # add text centered on image
        cv2.putText(img, text, (textX, textY), font, 1, (255, 255, 255), 2)

        if counter % detection_freq == 0:
            # load image to predict
            img = image.load_img(complete_path, target_size=image_size)
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = model_preprocess(img)

            probabilities = model.predict(img,
                                          batch_size=1,
                                          verbose=0,
                                          steps=None,
                                          callbacks=None,
                                          max_queue_size=10,
                                          workers=1,
                                          use_multiprocessing=False)[0]

            # transform [0,1] values into percentages and associate it to its class name
            result = [(classes[i], float(probabilities[i]) * 100.0) for i in range(nbr_classes)]

            # sort the result by percentage
            result.sort(reverse=True, key=lambda x: x[1])

            # get maximum probability and corresponding class
            max_class, max_proba = result[0][0], result[0][1]

            # load image for writing the video
            img = cv2.imread(complete_path)

            # convert image to RGB since we trained images using RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # setup text
            font = cv2.FONT_HERSHEY_SIMPLEX

            # probability is formatted to have two digits after the coma
            text = str(max_class) + " : " + str("{:.2f}".format(max_proba)) + "%"

            # get boundary of this text
            textsize = cv2.getTextSize(text, font, 1, 2)[0]

            # get coordinates based on boundary
            textX = (img.shape[1] - textsize[0]) // 2
            textY = (img.shape[0] + textsize[1]) // 2

            # add text centered on image
            cv2.putText(img, text, (textX, textY), font, 1, (255, 255, 255), 2)

        counter = counter + 1
        video_writer.append_data(img)

    video_writer.close()
