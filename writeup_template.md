**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/grayscale.png "Grayscaling"
[image3]: ./examples/random_noise.png "Random Noise"
[image4]: ./test/sign1.jpg "Traffic Sign 1"
[image5]: ./test/sign2.jpg "Traffic Sign 2"
[image6]: ./test/sign3.jpg "Traffic Sign 3"
[image7]: ./test/sign4.jpg "Traffic Sign 4"
[image8]: ./test/sign5.jpg "Traffic Sign 5"
[image9]: ./examples/layer_0.png "Sign"
[image10]: ./examples/layer_1.png "Layer 1"


## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
**Writeup / README**

You're reading it! and here is a link to my [project code](https://github.com/odats/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

**Data Set Summary & Exploration**

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is: 34799
* The size of the validation set is: 4410
* The size of test set is: 12630
* The shape of a traffic sign image is: (32,32,3)
* The number of unique classes/labels in the data set is: 43

Here is an exploratory visualization of the data set. It is a bar chart showing the train and test data. It also includes generated augmented signs images.

![alt text][image1]

**Design and Test a Model Architecture**

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because the color (rgb) does not make big impuct on result.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because to makes the system learn faster. Simplified math calculation.

I decided to generate additional data because it help to train system better. Additional train data increases accuracy.

To add more data to the the data set, I used the following techniques: blurred and sharpen.

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is more blurred and sharpen edges.


**Final model architecture**

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Normalize Input  		| 32x32x1 Grayscale image   					| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 					|
| Fully connected		| Input = 400. Output = 120.         			|
| RELU					|												|
| Dropouts				| 50%											|
| Fully connected		| Input = 120. Output = 84.        				|
| RELU					|												|
| Dropouts				| 50%											|
| Logits				| Input = 84. Output = 43.         				|
| Softmax				| To get cros entropy       					|
 
To train the model, I used:
*optimizer: AdamOptimizer
*batch size: 128
*number of epochs: 10
*learning rate: 0.001

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of: 0.953
* test set accuracy of: 0.940

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
I tried LeNet. It worked well from the beginning.

* What were some problems with the initial architecture?
I change input and output.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
I had overfitting and used dropouts to improve results.

* Which parameters were tuned? How were they adjusted and why?
I tried to tune different parameters but could not get huge improvements in accuracy. Only increasing training dataset improved final results.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
CNN work well becouse it was designed for similar problems. Dropout layer helped to resolve overfitting.

If a well known architecture was chosen:
* What architecture was chosen? 
LeNet
* Why did you believe it would be relevant to the traffic sign application?
It was designed for similar problems (image classification)
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
Running model on training, validation and test sets showed results higher than 93%
 

**Test a Model on New Images**

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

Most of signs should be classified without problems.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Ahead only   									| 
| Yield     			| Yield 										|
| Bumpy road			| Bumpy road 									|
| Priority road	  		| Priority road					 				|
| Keep left    			| Keep left      		  						|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 94%

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

6.71345 Ahead only
6.69856 Yield
2.03834 Priority road
1.15855 No passing
0.230938 No vehicles

108.606 Yield
19.0339 No vehicles
9.14488 Priority road
-1.477 Ahead only
-9.27126 Keep right

18.899 Bumpy road
13.5459 Bicycles crossing
2.95245 Stop
2.74061 Turn left ahead
2.31678 Ahead only

35.4564 Priority road
1.05222 Roundabout mandatory
-1.97664 End of all speed and passing limits
-3.1961 End of no passing
-6.56845 Speed limit (50km/h)

8.6213 Children crossing
7.22221 Ahead only
6.03431 Go straight or right
5.15688 Bicycles crossing
3.66011 Yield



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
First conv layer: simple shapes. I can guess a sign.
Second conv layer: complex shapes. Looks more like combination of pixels. I can not guess the sign.

![alt text][image9]

![alt text][image10]

