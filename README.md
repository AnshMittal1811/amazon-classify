# Viewing the Amazon From Space 

This repository contains a solution to the Kaggle Competition 'Viewing the Amazon From Space.' The code is based heavily on Jason Brownlee's post on how to build a CNN using the Planet data at this link. This README will cover the following steps of our own data pipeline. 
  
  * Introduction
  * How to Prepare Data for Modeling
  * Model Evaluation Measure
  * How to Evaluate a Baseline Model
  * How to Improve Model Performance
  * How to use Transfer Learning
  * How to Finalize the Model and Make Predictions
  
We will aim to use this as a template for future Sociometrik tasks using machine learning on satellite imagery.

# Introduction 

The Planet data consists of 40000 256 X 256 .jpg images along with 17 labels that were created and and labelled by the organization Planet. The testing data consists of another 80000 images that were used to grade the final solutions. These testing data are also 256 X 256 .jpg images. 

The target labels consist of different land types and descriptions which may occur together (hazy, desert, forest, rainy ,etc). There are 17 possible labels. This is a **multi-label classification problem.** This means that any subset of 17 labels can exist in a single picture. Our task is to predict which of these 17 labels exists in any given image of the Amazon rainforest.

# How to Prepare Data for Modeling

There are two steps in our process for preparing the data for modelling. First we have to process the images and then we have to process the labels so that both are structured in an easy way for us to train the model.

#### Clone the repository


#### Folder structure

The folder structure for the scripts is the same as in the GitHub repository except for the  training data folders. This is not stored on GitHub as it is too large and we do not want to put it into version control anyway. The structure diagram is given below: 
     
     --- planet-rep
	       --- train-jpg
		       --- train_0.jpg
		       --- train_1.jpg
		       --- train_2.jpg
		       ...
		       --- train_40478.jpg
	       --- train_v2.csv
	       --- planet_data.npz
	       --- planet_rep_train.ipynb
	       --- planet_rep.ipynb
	       --- README.md
	       --- requirements.txt
	       --- setup.sh
	       --- test_evaluation_plot.png
	       --- train-jpg.tar
	       --- train-jpg.tar.7z

#### Getting the training data

- Go to the Kaggle competition page for **Planet: Viewing the Amazon from Space** at this link. 
- Then click on the **data** tab. 
- Then scroll down until you find a folder called **train-jpg**. 
- Hover your mouse over this until a **download icon** appears.
- Click on the download icon 
- The download of train-jpg.tar should start - Once this is done you can use any program like Winzip or Mac Unarchiver to unzip the tar file
- You will then have the folder **train-jpg** with the jpg images    
- Move **train-jpg.tar** and **train-jpg** into the repository folder according to the structure above
- Then go back to the Kaggle competition page 
- Download **train_v2.csv** and place it in the folder according to the structure above


#### Images 

First we compress the images to 32 X 32 or any other desired target size and then convert these to a numerical array. We need to compress the images if we are running this without a GPU regardless of whether they are on our local machine or on a cloud server. This is to keep the training time manageable as this is just a test case for us to learn the data processing pipeline for satellite imagery. The function which does this in the code is: 

```python 
		def compress:
			...	
```

Just add a function call to this and it will be run as part of the pipe-line. It will output a file with format **.npz** which is a Python built-in compressed file format for numerical arrays to the repository folder.	
		 
#### Labels 

They are given separately from the images in the train_2.csv file with two columns: 1) file name and 2) label in the format 'haze rain desert forest.' This means each data point is labelled by writing in plain text the words corresponding to whatever exists in the picture in one string separated by single spaces. An example screenshot of the from the

Get unique class labels, create mapping dictionary between class labels and integers and vice versa, use these dictionaries to one-hot encode the class labels for each picture

# How to prepare hardware

# Model evaluation measure

The model evaluation measure is 

# Evaluate a base-line model


# Improve Model Performance

# How to use transfer learning


# How to finalise the model and make predictions 






# Our solution 


# Comparison to winner's solution 


# Conclusion
