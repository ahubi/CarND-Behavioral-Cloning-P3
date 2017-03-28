#**Behavioral Cloning**

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeupimages/cnn-architecture-624x890.png "Nvidia CNN Architecture"
[image2]: ./writeupimages/original.jpg "Center image"
[image3]: ./writeupimages/flipped.jpg "Center image flipped"
[image4]: ./writeupimages/cropped.jpg "Center image cropped"
[image5]: ./writeupimages/original2.jpg "Center image 2"
[image6]: ./writeupimages/flipped2.jpg "Center image 2 flipped"
[image7]: ./writeupimages/cropped2.jpg "Center image 2 cropped"
[image8]: ./writeupimages/fromleft2right.png "Left to right"
[image9]: ./writeupimages/fromright2left.png "Right to left"
[image10]: ./writeupimages/mse_model_loss_15_epochs.png "MSE Loss Diagram"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project is located on [github](https://github.com/ahubi/CarND-Behavioral-Cloning-P3) includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup.md containing writeup report
* run1.mp4 containing autonomous driving video of first track


####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 and 5x5 filter sizes and depths between 24 and 64 (model.py lines 87-96)

The model includes RELU layers to introduce nonlinearity (lines 87-91), and the data is normalized in the model using a Keras lambda layer (lines 79-81). Additionally cropping layer is added to remove irrelevant information from the input (line 84).

####2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (lines 69-73). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 97).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used  only center camera images and augmented every training image by rotating it. In total 10000 images were recorded, total size of about 200 MB.

For details about how I created the training data, see the next section.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with a simple model and try to train the model with it on sample data set.

My first step was to use a convolution neural network with just one layer and see how the car behaves.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting and bad autonomous behavior of the car I decided to use much powerful and mature architecture as described by Nvidia self driving team.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. Although track 1 can be considered as relatively easy there are still the following spots worth mentioning:

* Sharp and long curves
* Bridge with different pavements the the rest of track and different sites
* Road shoulders with no line marking, but exit spots
* Road shoulders with different line colors (white / grey)
* Shadows from the trees and overhanging cables

to improve the driving behavior in these cases, I tried to record more data of the tracks. Creation of training data and the training process is described in the next section.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. Please watch the recorded video to see the autonomous driving of the car.

####2. Final Model Architecture

The final model architecture (model.py lines 87-96) is based on Nvidia CNN Architecture for self driving cars. [Link](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/)

Here is a visualization of the architecture:

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I drove following laps and recorded training data using center lane driving:

* 2-3 laps normal driving.
* 1 lap recording driving away from the left or right site to teach the car how to recover when is comes close to the site
* 1 lap smooth curve driving
* 1 lap driving in opposite direction

Here is an example image of center lane driving:

![alt text][image5]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to how to return to the center of the road when it drives to the site. These images show what a recovery looks like starting from left:

![alt text][image8]

and here is starting from right:

![alt text][image9]


To augment the data sat, I also flipped images and angles thinking that this would the model more valuable input to train on. For example, here is an original image recorded by the center camera:

![alt text][image2]

and below is the flipped image:

![alt text][image3]

to remove irrelevant information from the pictures a cropping Keras layer is added to the model. Below is the corresponding image:

![alt text][image4]

Additionally a Lamda layer is added to for image normalization.
After the collection process, I had 10000 number of data points.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was starting with 6 as evidenced by the diagram of MSE model.

![alt text][image10]

I used an adam optimizer so that manually training the learning rate wasn't necessary.
