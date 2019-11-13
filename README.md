# Forest fire detection using CNN
This project is an attempt to use convolutional neural networks (CNN) to detect the presence or the start 
of a forest fire in a given picture. The idea is that this model could be applied to detect a fire or a start of 
a fire from (aerial) surveillance footage of a forest. The model could be applied in real-time to low-framerate surveillance
video (with fires not moving very fast, this assumption is somewhat sound) and give alert in case of fire.

# Required libraries
Keras, OpenCV, 

# Datasets
Our networks are trained on three datasets, each containing images of three categories : 'fire', 'no fire', 'start fire' totalling
at around 6000 images. These images are mostly of forest or forest-like environments. Images labelled 'fire' contain visible flames
somewhere in the image,  'start fire' images contain smoke indicating the start of a fire. Finaly, images labelled 'no fire' are
images taken in forests. 

In order to train a network which generalizes well to new images, we used data augmentation provided by Keras. 
This applies rotations, zooms and translations randomly to the training set. 

# Results
The result of our models, measured by categorical loss and accuracy is summarized in the following table.

The Inception-V3-based model has very good performance.

![](video_examples/video_0.gif)
![](video_examples/video_1.gif)
![](video_examples/video_2.gif)
![](video_examples/video_3.gif)
