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



## German Traffic Sign Classifier

This classifier has an architecture inspired by LeNet.

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

My final model results were:
* training set accuracy of 0.997
* validation set accuracy of 0.973
* test set accuracy of 0.955

I have run the classifier on these images found online.

| image 0	(Yield) | image 1	(Turn right ahead) | image 2 (Priority road) | image 3	(Speed limit 30km/h) | image 4 (Speed limit 120km/h) | image 5 (No passing)		         |
|:-----------------:|:-----------------:|:-----------------:|:-----------------:|:-----------------:|:-----------------:| 
| ![alt text][img0] | ![alt text][img1] | ![alt text][img2] | ![alt text][img3] | ![alt text][img4] | ![alt text][img5] |

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

These are the top five probabilities of predictions on each emage

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


### Dependencies
This program requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)


### Dataset and Repository

* Clone the project, which contains the data.
```sh
git clone https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project
cd CarND-Traffic-Sign-Classifier-Project
jupyter notebook Traffic_Sign_Classifier.ipynb
```
