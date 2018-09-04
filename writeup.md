# **Traffic Sign Recognition** 

## Writeup


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)
[train_hist]: ./writeup_images/train_hist.png 
[valid_hist]: ./writeup_images/valid_hist.png 
[test_hist]: ./writeup_images/test_hist.png
[augmented_hist]: ./writeup_images/augmented_hist.png

[original]: ./writeup_images/original.png "original image"
[gray]: ./writeup_images/gray.png "grayscale image"
[CLAHE]: ./writeup_images/CLAHE.png "CLAHE image"
[normalized]: ./writeup_images/normalized.png "normalized image"
[augmented1]: ./writeup_images/augmented1.png "augmented image 1"
[augmented2]: ./writeup_images/augmented2.png "augmented image 2"
[augmented3]: ./writeup_images/augmented3.png "augmented image 3"

[img0]: ./writeup_images/img0.jpg
[img1]: ./writeup_images/img1.jpg
[img2]: ./writeup_images/img2.jpg
[img3]: ./writeup_images/img3.jpg
[img4]: ./writeup_images/img4.jpg
[img5]: ./writeup_images/img5.jpg

[bar_chart]: ./writeup_images/bar_chart.png

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/Kazutaka333/traffic_sign_classifier/blob/master/Traffic_Sign_Classifier.ipynbhttps://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here are histograms of training, validation, test set categorized into each sign class.



![training histogram][train_hist]  ![validation histogram][valid_hist] ![test histogram][test_hist] 
 




### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because that reduces the large amount of training time without sacrificing the accuracy of the classifier. Contrast limited adaptive histogram equalization (CLAHE) gives more vivid contrast in a image, which, I think, helps the model to recognize the line and curves better. I've chosen the division by 255 as my normalization and it actually showed the drasctic improvment on accuracy.

Here is an example of each proccessed image.


![alt text][original] | ![alt text][gray] | ![alt_text][normalized] |![alt_text][CLAHE]
:--------------------:|:-----------------:|:-----------------------:|:-----------------:
 | | |

To obtain the better accuracy, I've created augmented data from the training set. For preventing fake data from being too much noise for the model, the augmentation process needed to be one that alter the original image a little bit and randomly and I've added three ways of image modification; warping, zooming, and, tilting. Augmented data is created from applying those three methods by a random small extent.

Here are a couple of examples of an augmented image:


![alt text][augmented1]  ![alt text][augmented2]  ![alt text][augmented3] 


The difference between the original data set and the augmented data set is shown in the following histograms.
I created augmented data for each class in a way that the number of images in each class does not exceed the original maximum number of images in one class so that the number of fake data is not too much that dominates the model.

![alt text][train_hist]  ![alt text][augmented_hist] 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer          		|     Description																	         					| 
|:----------------:|:---------------------------------------------:| 
| Input          		| 32x32x1 grayscale image  																					| 
| Convolution 5x5 	| 1x1 stride, valid padding, outputs 28x28x6 	  |
| RELU					        |																																															|
| Max pooling	    	| 2x2 stride,  outputs 14x14x6  																|
| Convolution 5x5	 | 1x1 stride, valid padding, outputs 10x10x16			|
| RELU					        |																																															|
| Max pooling	    	| 2x2 stride,  outputs 5x5x16   																|
| Fully connected		| outputs 120                                   |
| RELU					        |																																															|
| Dropout          | 50% keep probability                          |
| Fully connected		| outputs 84                                    |
| RELU					        |																																															|
| Fully connected		| outputs 43                                    |
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I did not changed so many parameters from the LeNet implementation in the course. The batch size, optimizer, and learning rate are 128, Adam optimizer, and 0.001, which are given in the course. However, I changed epochs from 20 to 30, which seemed to yeild better accuracy with not too much training time.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.997
* validation set accuracy of 0.973
* test set accuracy of 0.955

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

Since I was already comfortable with LeNet from the previous quiz, I've decided to try it out on this project too.

* What were some problems with the initial architecture?

With the initial architecture, the accuracy was about 80%, which was not as high as 93%. This architecture tended to overfit.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

Since I knew the number of images in each class was not even and simply more different images should improve the model, the augmented images were added in the manner explained above. From the gap between validation accuracy and training accuracy, I added dropout layer to avoid overfitting. In fact, I also tried the architecture on the provided research paper (Traffic Sign Recognition with Multi-Scale Convolutional Networks by Pierre Sermanet and Yann LeCun) but the validation accuracy never went up above 86%. After that, I made a lot of small tweak on the augmentation method. At the end, changing the normalization from the one provided in the note to the division by 255(÷255) shows drastic improvements on the validation accuracy. I was not sure why this happned and another student also claimed the same thing happened. The mentor I asked was not sure either. 

* Which parameters were tuned? How were they adjusted and why?

I increased epoch to gain more accuracy. I increased the number of augmented data way more than I have right now to see if it shows any improvement.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
 
Convolutional layer can extract a certain feature in an image. Each traffic sign that is meant to be not to mistaken by human has very distinctive visual feature. This is why convnet works well with this problem. Dropout and augmented data regularize the model and prevent overfitting. However, I have to say the thing that prevents me from submitting a project was definitely the normalization.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

| image 0	(Yield) | image 1	(Turn right ahead) | image 2 (Priority road) |
|:-----------------:|:-----------------:|:-----------------:|
| ![alt text][img0] | ![alt text][img1] | ![alt text][img2] |
| I think this image could be pretty easy to classify due to the vivid contrast between the background and the sign | This sign also has good separation from the background and it's the almost ideal image of the sign | this picture is a little darker than other pictures and the part of a house is in the picture, which might affect the classifier accuracy |

| image 3	(Speed limit 30km/h) | image 4 (Speed limit 120km/h) | image 5 (No passing) |
|:-----------------:|:-----------------:|:-----------------:|
| ![alt text][img3] | ![alt text][img4] | ![alt text][img5] |
| This picture is the smallest among these 6 but it's not really an issue since each picture is fed into the classifier after resizing into 32x32 | This picture has the nice plain background as well as image 1. The traffic sign with speed limit number could be more challenging since there are speed limit sign with other numbers as well. | This image has the very strong contrast in the background which might interfere with the result of the classifier. |


Overall, the images has the straight angle and brightness and the signs are neither damaged nor covered with something. I think they are fair game for this classifier.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			              |     Prediction	 					| 
|:---------------------:|:--------------------:| 
| Yield          		     | Yield   							    		| 
| Turn right ahead      | Turn right ahead 				|
| Priority road         | Priority road						  |
| Speed limit 30km/h    | Speed limit 30km/h			|
| Speed limit 120km/h   | Speed limit 120km/h 	|
| No passing            | No passing           | 


The model was able to correctly guess 6 of the 6 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of accuracy 95.5%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 20th cell of the Ipython notebook.

*image 0*

| Probability         	|     Prediction	        					| 
|:--------------------:|:---------------------------:| 
| 1.00 (lost precision)| Yield   									| 
| 8.28896e-14 	        | No vehicles						| 
|	5.02711e-15 	        | Priority road						| 
|	4.1249e-15 	         | Stop						| 
|	7.78334e-16 	        | Keep right						| 

*image 1*

| Probability         	|     Prediction	        					| 
|:--------------------:|:---------------------------:| 
| 0.999994    | 	 Turn right ahead |
|	5.67283e-06 |	 Ahead only |
|	6.09594e-10 |	 No vehicles |
|	4.83902e-10 |	 Keep left |
| 2.11305e-10 |	 Roundabout mandatory |

*image 2*

| Probability         	|     Prediction	        					| 
|:--------------------:|:---------------------------:| 
| 1.0 (lost precision)  |   	 Priority road |
| 8.71962e-09 	|  Stop |
| 2.33766e-09 	|  No vehicles |
| 7.9627e-10 	 | End of all speed and passing limits |
| 3.52491e-10 	|  No passing |
  
*image 3*

| Probability         	|     Prediction	        					| 
|:--------------------:|:---------------------------:| 
| 0.998181    | 	 Speed limit (30km/h) |
| 0.0010598   | 	 Speed limit (80km/h) |
| 0.000508322 |   	 Speed limit (50km/h) |
| 0.000196236 |  	 End of speed limit (80km/h) |
| 3.89811e-05 | 	 Speed limit (20km/h) |
  

*image 4*

| Probability         	|     Prediction	        					| 
|:--------------------:|:---------------------------:| 
| 0.7852     |  	 Speed limit (120km/h)   |
|	0.10838    |  	 Speed limit (70km/h)   |
|	0.10636    |  	 Speed limit (20km/h)   |
|	4.01502e-05|  	 Speed limit (80km/h)   |
|	1.75141e-05|  	 Speed limit (100km/h)   |

Even though the prediction is correct on image 4, the model considers a little bit of possibility that the sign could be Speed limit (70km/h) or Speed limit (20km/h).

  
*image 5*

| Probability         	|     Prediction	        					| 
|:--------------------:|:---------------------------:| 
| 0.999999    | 	 No passing |
| 7.57351e-07 |	 Slippery road |
| 1.13976e-07 |	 No passing for vehicles over 3.5 metric tons |
| 1.96753e-08 |	 End of no passing |
| 4.60882e-09 |	 Dangerous curve to the right |

![][bar_chart]
