import argparse
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from evaluate_model import evaluate_model, extract_hard_samples
from transfer_learning import train_simpler_inception_based_model
from video_annotation import video_fire_detection
from keras.applications.inception_v3 import preprocess_input as inception_preprocess_input


if __name__ == '__main__':

    classes = ['fire', 'no_fire', 'start_fire']

    parser = argparse.ArgumentParser(description='Convolutional neural network for forest fire detection',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    subparsers = parser.add_subparsers(title='',
                                       description='Network can be trained on a provided dataset or predictions can be'
                                                   'made using a pre-trained model. Models can also be evaluated.',
                                       help='', dest='mode')

    subparsers.required = True

    parser_train = subparsers.add_parser('train',
                                         help='Create and train the simpler InceptionV3-based model.',
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

    parser_train.add_argument('-freeze',
                              type=bool,
                              action='store',
                              dest='freeze',
                              help='Whether to freeze every layer except the last fully connected ones.',
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

    parser_tune = subparsers.add_parser('tune', help='Fine-tune a pre-trained Inception-V3-based model.',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

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

    parser_tune.add_argument('-freeze',
                             type=bool,
                             action='store',
                             dest='freeze',
                             help='Whether to freeze every layer except the last fully connected ones.',
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
                              help='Path to output the annotated mp4 video.',
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
                              type=int,
                              action='store',
                              dest='freq',
                              help='Prediction is to be made every freq frames.',
                              default=12,
                              required=False)

    parser_extract = subparsers.add_parser('extract',
                                           help='Extract hard examples from a dataset (samples classified with low '
                                                'confidence).')

    parser_extract.add_argument('-data',
                                type=str,
                                action='store',
                                dest='dataset',
                                help='Path to a dataset.',
                                default=argparse.SUPPRESS,
                                required=True)

    parser_extract.add_argument('-model',
                                type=str,
                                action='store',
                                dest='model_path',
                                help='Path to a trained model.',
                                default=argparse.SUPPRESS,
                                required=True)

    parser_extract.add_argument('-threshold',
                                type=float,
                                action='store',
                                dest='extract_threshold',
                                help='Threshold for the hard examples.',
                                default=argparse.SUPPRESS,
                                required=True)

    parser_test = subparsers.add_parser('test',
                                        help='Test a model on a test set of images.')

    parser_test.add_argument('-data',
                             type=str,
                             action='store',
                             dest='dataset',
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

        train_simpler_inception_based_model(parsed.dataset,
                                            fine_tune_existing=None,
                                            freeze=parsed.freeze,
                                            learning_rate=0.001,
                                            percentage=parsed.proportion,
                                            nbr_epochs=parsed.epochs,
                                            batch_size=parsed.batch_size)

    elif parsed.mode == "tune":

        train_simpler_inception_based_model(parsed.dataset,
                                            fine_tune_existing=parsed.model_path,
                                            freeze=parsed.freeze,
                                            learning_rate=parsed.learning_rate,
                                            percentage=parsed.proportion,
                                            nbr_epochs=parsed.epochs,
                                            batch_size=parsed.batch_size)

    elif parsed.mode == "predict":

        model = load_model(parsed.model_path)

        img = image.load_img(parsed.image_path, target_size=(224, 224, 3))

        # processed image to feed the network
        processed_img = image.img_to_array(img)
        processed_img = np.expand_dims(processed_img, axis=0)
        processed_img = inception_preprocess_input(processed_img)

        # get prediction using the network
        predictions = model.predict(processed_img)[0]

        print(predictions)

    elif parsed.mode == "video":

        video_fire_detection(parsed.input_video_path,
                             parsed.output_video_path,
                             parsed.model_path,
                             inception_preprocess_input,
                             (224, 224),
                             parsed.freq)

    elif parsed.mode == "extract":
        print(extract_hard_samples(parsed.model_path,
                                   inception_preprocess_input,
                                   parsed.dataset,
                                   parsed.extract_threshold))

    elif parsed.mode == "test":
        print(evaluate_model(parsed.model_path,
                             classes,
                             inception_preprocess_input,
                             parsed.dataset))
