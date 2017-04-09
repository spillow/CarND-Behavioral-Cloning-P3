
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/arch_viz.png "Model Visualization"
[image2]: ./examples/center_example.jpg "Center Driving"
[image3]: ./examples/center_from_left.jpg "Recovery From Left"
[image4]: ./examples/advanced_return_from_left.jpg "Advanced Recovery From Left"
[image5]: ./examples/return_from_right.jpg "Recovery From Right"

---

The project includes the following files:
* [model.py](https://github.com/spillow/CarND-Behavioral-Cloning-P3/blob/master/model.py) containing the script to create and train the model
* [drive.py](https://github.com/spillow/CarND-Behavioral-Cloning-P3/blob/master/drive.py) for driving the car in autonomous mode
* model.h5 containing a trained convolutional/fully connected neural network
* [video.mp4](https://github.com/spillow/CarND-Behavioral-Cloning-P3/blob/master/video.mp4) which shows the car driving around track 1.
* [advanced.mp4](https://github.com/spillow/CarND-Behavioral-Cloning-P3/blob/master/advanced.mp4) A partial drive around the more hilly track
* writeup_report.md

***2. Submission includes functional code***
Using the Udacity provided simulator and the drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

***3. Submission code is usable and readable***

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline used for training and validating the model (with the architecture defined in define_model()), and it contains comments to explain how the code works.

**Model Architecture and Training Strategy**

***1. An appropriate model architecture has been employed***

The model consists of a convolution neural network with 5x5 filter sizes and depths between 3 and 80 (model.py lines 50-59).

The model includes RELU layers to introduce nonlinearity (code line 53-67), and the data is normalized in the model using a Keras lambda layer (code line 48).

***2. Attempts to reduce overfitting in the model***

The model contains maxpooling layers after every convolutional layer in order to reduce overfitting (model.py lines 54-60).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 99). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

***3. Model parameter tuning***

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 110).

***4. Appropriate training data***

Training data was chosen to keep the vehicle driving on the road. A combination of center lane driving, recovering from the left and right sides of the road, driving backwards through the track, generalizing using the advanced track, and repeated drivebys from different angles on difficult sections of the track.

For details about training data collection, see the next section.

**Model Architecture and Training Strategy**

***1. Solution Design Approach***

The overall strategy for deriving a model architecture was to start with an architecture that has generally worked
well on other image recognition tasks.

The first step was to use a convolution neural network model similar to LeNet/AlexNet/etc.  The general approach is:

| Architecture               |
|:--------------------------:|
| Convolution                |
| Max Pooling                |
| ...                        |
| Flatten                    |
| # of Fully Connected Layers|

The standard approach of partitioning the data into a training and validation set was used.  The initial architecture
used had very similar validation and training mean squared error loss so overfitting wasn't an issue.  Having found
that adding more output channels in each convolutional layer worked with the traffic sign dataset, more were added here
with some tweaking of the size and number of fully connected layers to arrive at the final architecture.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. For example, the early models had difficulty in sections where there was dirt alongside the road
instead of lines or the striped patches.  Considering the network architecture was strong enough to drive the rest of the track,
the issue seemed to be with insufficient training data in those sections.  To improve the driving behavior in these cases,
these sections of road were driven to and "record" was alternately turned off and on while approaching from different angles
and from more centered or off the edge of the road.

At the end of the process, the vehicle learned how to deal with those difficult patches and is able to drive autonomously around the track without leaving the road.

***2. Final Model Architecture***

The final model architecture (model.py lines 46-69) consists of a convolution neural network with the following layers and layer sizes:

| Layer         		    |     Description	        					            |
|:---------------------:|:---------------------------------------------:|
| Input         		    | 320x16x3 RGB image   							            |
| Lambda                | Image normalization  							            |
| Convolution 1x1x3   	| 1x1 stride, valid padding, outputs 320x160x3  |
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 316x156x6 	|
| RELU					        |												                        |
| Max Pooling           |	2x2 stride,	valid padding, outputs 158x78x6   |
| Convolution 5x5    	  | 1x1 stride, valid padding, outputs 154x74x20 	|
| RELU					        |												                        |
| Max Pooling           |	2x2 stride,	valid padding, outputs 77x37x20   |
| Convolution 5x5    	  | 1x1 stride, valid padding, outputs 73x33x40 	|
| RELU					        |												                        |
| Max Pooling           |	2x2 stride,	valid padding, outputs 36x16x40   |
| Convolution 5x5    	  | 1x1 stride, valid padding, outputs 32x12x80	  |
| RELU					        |												                        |
| Max Pooling           |	2x2 stride,	valid padding, outputs 16x6x80    |
| Flatten               |	outputs 7680					                        |
| Fully connected		    | outputs 120                                   |
| RELU					        |												                        |
| Fully connected		    | outputs 84                                    |
| RELU					        |												                        |
| Fully connected		    | outputs 40                                    |
| RELU					        |												                        |
| Fully connected		    | outputs 20                                    |
| RELU					        |												                        |
| Fully connected		    | outputs 10                                    |
| RELU					        |												                        |
| Fully connected		    | outputs 1                                     |

Here is a visualization of the architecture (minus the color space converter):

![alt text][image1]

This figure is generated by adapting the code from https://github.com/gwding/draw_convnet

***3. Creation of the Training Set & Training Process***

To capture good driving behavior, three laps on track one were recorded using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

The vehicle was then recorded recovering from the left side and right sides of the road back to center so that the vehicle would learn to turn the steering wheel the proper direction from those positions.  It can only know what it has been taught.
These images show what a recovery looks:

![alt text][image3]
![alt text][image4]
![alt text][image5]

The process was repeated on track two in order to get more data points.

After this data collection process, no data set augmentation (such as image flipping or left/right cameras) was needed.

After the collection process, the dataset size was 18510 images. The data was preprocessed by normalizing it in the range
[-0.5, 0.5].

The data was randomly shuffled and 20% went into the validation set.

This data was used for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the fact that the MSE loss roughly bottomed out at that point. The adam optimizer was used so that manually training the learning rate wasn't necessary.

To train the model, any number of directories from different recording runs can be used.  For example:

```sh
python model.py records\center_driving records\return_to_center records\return_to_center_from_left
```

The --visualize option may be used to [output](https://github.com/spillow/CarND-Behavioral-Cloning-P3/blob/master/examples/model.png) a graphviz representation of the network.
