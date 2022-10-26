# American-Sign-Language-Detection
American Sign Language Detection using Neural Networks and Open CV and other pretrained models

1.	Research Scenario and My Research Question:
This Project aims at detecting American Sign Language through CNN models and then focussing on its detection in real time camera using Computer Vision
2.	Overview of the Data:
The Dataset can be downloaded from Kaggle using below link:
https://www.kaggle.com/datasets/grassknoted/asl-alphabet
The data set (ASL Alphabet Train)is a collection of images of alphabets from the American Sign Language, separated in 29 folders which represent the various classes. The Dataset contains 29 Classes (26 Alphabets and Space, Del and Nothing) and contains 3000 Images for each classes and total of 87000 Images of 200*200 Pixels

3.	Data Visualization:
The dataset is balanced and same can be viewed from the visualisation below:

 

 

4.	Data Pre-processing:
There were 29 Classes each with different folder and 3000 Images, so to make a csv out of the filepath and the labels for them to access them and to read and process on them, then transformed the image to 224*224 and then used image augmentation to generate images and make image generators from the same to make train_images, val_images and test_images.

First, Train and Test split was done for test_ratio being 0.2 and then val_images being 0.25 of the train images. Count of the images is shown below in each generator
 
5.	Transfer Learning CNN model:

Used different pre-trained model and ran for one epoch to decide on the best pre-trained model for the image dataset and chose the best for building the CNN model. The result for one epoch on the image dataset was as follows:
 

  
Since the best val_accuracy after one epoch was given by DenseNet201, hence went ahead to build the transfer learned model.
1.  Design Details:

Defined the input_shape to be (224,224) and mentioned the pre-trained model trainable to be false.
Added two dense layer with 128 and activation function as relu and the outermost dense layer with 29 and with Softmax Function as the activation function.
Compiled the model with “adam” optimizer and ” categorical_crossentropy” and with metrics as accuracy.
CNN model was then fitted on the training set and validation set with 5 epochs and callback with early_stopping monitored at val_accuracy and patience being 1 and restoring the best weights to be true and then evaluated on the test_images


2. Data Generator, Augmentation and Fine Tuning of Hyper Parameters:

Data generator was used to form the images of batch size = 32 and zoom_range=0.15, width_shift_range=0.2,  height_shift_range=0.2,shear_range=0.15, horizontal_flip=False, fill_mode="nearest" for the train images with validation_ratio set to 0.25 and same was also followed for the test images and the following were the number of images generated for the train_images, Validation_images and the test-images respectively.
 

3. Model Evaluation:

With Augmentation
 


The Accuracy of transfer_learned model on the test dataset with augmentation was 98.48%
 
Without Augmentation:
 
The Accuracy of transfer_learned model on the test dataset without augmentation was 92.85%

 

 



6.	Designed CNN Model for Detection:
CNN model was designed to test on the test_images.
1. Design Details:

CNN model was designed as follows:
Convolution Layers having Kernel size 32 and (5,5) and input_shape as (224,224,3) and activation function as Relu and maxpooling layer of 2.
Two more convolution Layers having Kernel size 64 and (3,3) and input_shape as (224,224,3) and activation function as Relu and maxpooling layer of 2 were added.
Flatten layer was added followed by the Dense layer of 128 and activation function as Relu and then followe by the Dense Layer of 29 and activation function as Softmax.
Compiled the model with “adam” optimizer and ” categorical_crossentropy” and with metrics as accuracy.
CNN model was then fitted on the training set and validation set with 5 epochs and callback with early_stopping monitored at val_accuracy and patience being 1 and restoring the best weights to be true and then evaluated on the test_images

2. Data Generator, Augmentation and Fine Tuning of Hyper Parameters:
Data generator was used to form the images of batch size = 32 and zoom_range=0.15, width_shift_range=0.2,  height_shift_range=0.2,shear_range=0.15, horizontal_flip=False, fill_mode="nearest" for the train images with validation_ratio set to 0.25 and same was also followed for the test images and the following were the number of images generated for the train_images, Validation_images and the test-images respectively.
 
3. Model Evaluation:

With Augmentation

 

The Accuracy of Custom model  on the test dataset that I had achieved with data Augmentation was 97.72%
 
Without Data Augmentation:
 
The Accuracy of Custom model  on the test dataset that I had achieved without data Augmentation was 94.74 %
 
 

7.	Implementing ASL Detection in Real Time through Webcam

Computer Vision was used to implement real time detection of American Sign Language. CV2 Videocapture opens the base webcam of the Laptop and then using the model designed we can make the model and load weights and then do the detection.
Image Area was defined for the area and then the video was captured and the frames were extracted from the video and then the captured frame in the hand area is used for the detection of the hand signal and then the detected label is written on the video.
Since the functionalities of OpenCV not supported by Google Colab and Kaggle, had to implement this section on the personal system not having GPU with the saved model and weights.
8.	Results:
CNN model with the pre-trained model performed better than the custom model. The best pre-trained models out of the 10 Pre-trained models was utilised for the detection and got the accuracy of 98.87% while the custom model could only get the accuracy of 97.72% with data augmentation. 
When done without augmentation the best pre-trained models out of the 10 Pre-trained models was utilised for the detection and got the accuracy of 92.85% while the custom model could only get the accuracy of 94.74% with data augmentation.
When the model was tested in real_time it performed pretty well with some errors is the detection.

9.	Conclusion

The model was successfully built on the ASL dataset, and the real time detection of the American Sign language was possible. The accuracy achieved by both the model was more than 95%, with pre-trained achieving the higher accuracy with data Augmentation. Without data augmentation both the models got the accuracy above 90%. There were few shortcomings as well with unable to implement on the videos due to shortage of the necessary resources and CV2.Videocapture and many functionality of OpenCv not being supported by Kaggle and Google Colab

10.	References:

http://blog.leapmotion.com/getting-started-leap-motion-sdk/
https://www.researchgate.net/figure/ReLU-activation-function_fig3_319235847
https://www.kaggle.com/datamunge/sign-language-mnist#sign_mnist_test.zip
https://data.mendeley.com/datasets/c7zmhcfnyd/1Li, CC., Wu, MY., Sun, YC. (2021). 

11.	Code:

PLEASE FIND LINK TO MY COLAB NOTEBOOK HERE: 
https://colab.research.google.com/drive/11ZgY_JOsPEkkca6bcoUlH8ylSHoLIGV7?usp=sharing

