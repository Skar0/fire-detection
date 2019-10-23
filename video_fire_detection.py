import os
import cv2
import imageio
import numpy as np
from keras.engine.saving import load_model
from keras.preprocessing import image


def extract_images_from_video(video_path, images_directory):
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
            img_name = "frame_" + str(frame_nbr) + ".png"
            img_path = images_directory + img_name
            cv2.imwrite(img_path, frame)
            frame_nbr = frame_nbr + 1
        else:
            break

    # free video
    video.release()


def detect_fire(input_video_path, output_video_path, model_path, model_preprocess, image_size, detection_freq):
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
        print("--------------------------------------------- " + str(counter))
        complete_path = 'video_frames/' + img_path
        frames.append(complete_path)

        print(counter)
        print(complete_path)
        # load image for writing the video
        img = cv2.imread(complete_path)
        height, width, channels = img.shape

        # convert image to RGB since we trained images using RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # setup text
        font = cv2.FONT_HERSHEY_SIMPLEX

        text = str(max_class) + " : " + str(max_proba) + "%"

        # get boundary of this text
        textsize = cv2.getTextSize(text, font, 1, 2)[0]

        # get coords based on boundary
        textX = (img.shape[1] - textsize[0]) // 2
        textY = (img.shape[0] + textsize[1]) // 2

        print(img, text, (textX, textY), font, 1, (255, 255, 255), 2)
        # add text centered on image
        cv2.putText(img, text, (textX, textY), font, 1, (255, 255, 255), 2)

        if counter % detection_freq == 0:
            # load image to predict
            img = image.load_img(complete_path, target_size=image_size)
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = model_preprocess(img)

            probabilities = model.predict(img, batch_size=1, verbose=0, steps=None, callbacks=None, max_queue_size=10,
                                          workers=1, use_multiprocessing=False)[0]

            # transform [0,1] values into percentages and associate it to its class name (class_name order was used to
            # one-hot encode the classes)
            result = [(classes[i], float(probabilities[i]) * 100.0) for i in range(nbr_classes)]
            # sort the result by percentage
            result.sort(reverse=True, key=lambda x: x[1])

            max_class, max_proba = result[0][0], result[0][1]

            # load image for writing the video
            img = cv2.imread(complete_path)
            height, width, channels = img.shape

            # convert image to RGB since we trained images using RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # setup text
            font = cv2.FONT_HERSHEY_SIMPLEX

            # proba is formated to have two digits after the coma
            text = str(max_class) + " : " + str("{:.2f}".format(max_proba)) + "%"

            # get boundary of this text
            textsize = cv2.getTextSize(text, font, 1, 2)[0]

            # get coords based on boundary
            textX = (img.shape[1] - textsize[0]) // 2
            textY = (img.shape[0] + textsize[1]) // 2

            # add text centered on image
            cv2.putText(img, text, (textX, textY), font, 1, (255, 255, 255), 2)

        counter = counter + 1
        video_writer.append_data(img)

    video_writer.close()
