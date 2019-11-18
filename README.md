# Forest fire detection using CNN
This project is an attempt to use convolutional neural networks (CNN) to detect the presence or the start 
of a forest fire in an image. The idea is that this model could be applied to detect a fire or a start of 
a fire from (aerial) surveillance footage of a forest. The model could be applied in real-time to low-framerate surveillance
video (with fires not moving very fast, this assumption is somewhat sound) and give alert in case of fire. 

A description of the project, along with examples of video annotation by our network is provided below.

<img src="https://github.com/Skar0/fire-detection/blob/master/video_examples/video_0.gif" width="300">      <img src="https://github.com/Skar0/fire-detection/blob/master/video_examples/video_1.gif" width="300">


## Datasets
#### Provided dataset

Our network is trained on a provided dataset which contains images of three categories : 'fire', 'no fire', 'start fire' totalling around 6000 images. These images are mostly of forest or forest-like environments. Images labelled 'fire' contain visible flames,  'start fire' images contain smoke indicating the start of a fire. Finaly, images labelled 'no fire' are
images taken in forests. 

#### Augmenting the dataset

Our experiments showed that the network had trouble classifying 'start fire' images so we added images of this category to the dataset by extracting frames from videos showing the start of a fire. In order to train a network which generalizes well to new images, we used data augmentation functions provided by Keras to perform a series of random transformations (zooms, shifts, crops and rotations) on images before they are fed to the network.

## Project structure
Our goal was to create a legible project which handles every aspect of CNN creation and training. The code is organized as follows :

```bash
├── launcher.py
├── transfer_learning.py
├── video_annotation.py
├── evaluate_model.py
├── transfer_learned_model.h5
├── setup/
│   ├── setup_datasets.py
│   └── naive_approach.py
├── notebooks/
│   ├── datasets_analysis.ipynb
│   └── performance_analysis.ipynb
├── custom_model/
│   ├── cladoh.py
│   ├── model_train_test.py
│   └── model_train_tester.py
├── video_examples/
│   ├── video_0.gif
│   ├── video_1.gif
│   ├── video_2.gif
│   └── video_3.gif
```

The datasets can be setup using functions defines in setup_datasets.py. The model we used which performs transfer learning from InceptionV3 is defined in transfer_learning.py, this module contains a function that defines a batch generator which performs data augmentation. The training process is also handled in this file, with the possibility of freezing layers and adapting the learning rate for fine-tuning. Modules video_annotation.py allows to annotate a video with predictions from our CNN and evaluate_model.py allows us to evaluate our model and mine difficult examples for the network.

## Usage

#### Requirements:
The project was tested with the following versions of librairies:

      imageio==2.6.1
      imageio-ffmpeg==0.3.0
      Keras==2.1.6
      matplotlib==2.2.3
      numpy==1.15.1
      opencv-contrib-python==3.4.0.12
      Pillow==5.2.0
      tensorflow==1.5.1
   
#### Launcher

The module launcher.py contains a command-line parser which allows to launch our project.

##### Training

DATASET is the path to the dataset. PROPORTION must be in [0, 1] is the proportion of the dataset to be used for training, the rest is kept for validation. FREEZE is a boolean on whether to freeze the inception layers in the network. EPOCHS and BATCH_SIZE can be specified, their default values are 10 and 32.

    launcher.py train [-h] -data DATASET -prop PROPORTION -freeze FREEZE [-epochs EPOCHS] [-batch BATCH_SIZE]

##### Fine tuning

MODEL_PATH is the path to a pre-trained model, this must be a file containing wieghts + architecture. LEARNING_RATE can be specified, its default value is 0.001 (0.01 is used when training and is the default value of the optimizer we used).

    launcher.py tune [-h] -model MODEL_PATH [-lr LEARNING_RATE] -data DATASET -prop PROPORTION -freeze FREEZE [-epochs EPOCHS] [-batch BATCH_SIZE]

##### Perform a prediction

    launcher.py predict [-h] -path IMAGE_PATH -model MODEL_PATH
    
##### Extract difficult examples

Extracts images which are hard to classify by the model. When the network performs a prediction with a confidence level (probability) lower than EXTRACT_THRESHOLD for the correct class of an image, the path to this image is yielded.

    launcher.py extract [-h] -data DATASET -model MODEL_PATH -threshold EXTRACT_THRESHOLD
  
##### Metrics on a test set

Yields the metrics of our model on a test set.

    launcher.py test [-h] -data DATASET -model MODEL_PATH

##### Video annotation

The video given by INPUT_VIDEO_PATH is processed and prediction is performed on its frames, the annotated video is written to OUTPUT_VIDEO_PATH. A FREQ can be given so that only one out of every FREQ frames is extracted for prediction to speed up processing.

    launcher.py video [-h] -in INPUT_VIDEO_PATH -out OUTPUT_VIDEO_PATH -model MODEL_PATH [-freq FREQ]

## Results

#### Trained model file
Our trained model file containing the model architecture and trained weights is the file transfer_learned_model.h5 at the root of the project.

#### Performance
The performance of our model, measured by categorical loss and accuracy is the following:

On the provided test set :

    transfer_learned_model.h5
    100 samples
    loss : 0.3496805104602239 | acc : 0.9108910891089109 
    
On the whole dataset :

    transfer_learned_model.h5
    5953 samples
    loss : 0.0360565472869205 | acc : 0.9914328909793382
    
From our experiments, it seems that 'fire' and 'no fire' images are always lassified with high accuracy. Images labelled 'start fire' are harder to classify for the network. This may be explained by the fact that 'fire' images may contain smoke and that 'start fire' images sometimes contain small flames.
    
#### Video examples

Examples of videos annotated by our model can be found below.

<img src="https://github.com/Skar0/fire-detection/blob/master/video_examples/video_1.gif" width="350">
<img src="https://github.com/Skar0/fire-detection/blob/master/video_examples/video_2.gif" width="350">
<img src="https://github.com/Skar0/fire-detection/blob/master/video_examples/video_3.gif" width="350">
<img src="https://github.com/Skar0/fire-detection/blob/master/video_examples/video_0.gif" width="350">

