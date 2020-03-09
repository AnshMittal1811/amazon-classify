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

First clone the repository onto your local machine. The command to do this is given below: 

```shell
   git clone https://github.com/sociometrik/planet-rep.git 
```

This will create a folder called **planet-rep** on your machine with the Jupyter notebook scripts and others.

#### Folder structure

The folder structure for the scripts is the same as in the GitHub repository except for the  training data folders. This is not stored on GitHub as it is too large and we do not want to put it into version control anyway. The structure diagram is given below: 
     
     --- planet-rep
	       -- train-jpg
		       -- train_0.jpg
		       -- train_1.jpg
		       -- train_2.jpg
			       ...
		       -- train_40478.jpg
	       -- train_v2.csv
	       -- planet_data.npz
	       -- planet_rep_train.ipynb
	       -- planet_rep.ipynb
	       -- README.md
	       -- requirements.txt
	       -- setup.sh
	       -- test_evaluation_plot.png
	       -- train-jpg.tar
	       -- train-jpg.tar.7z

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


#### Labels 

Labels are given separately from the images in the train_2.csv file with two columns: 1) file name and 2) label in the format 'haze rain desert forest.' This means each data point is labelled by writing in plain text the words corresponding to whatever exists in the picture in one string separated by single spaces. To see what it looks like, we load the .csv file using the following function: 
```python
   def load_mapping_data('./train_v2.csv'):
	   ...
	   return(...)
```
Visualizing the first few rows of the returned data-frame gives us the following output: 
```
  
image_name 	tags 
0 train_0 	haze primary 
1 train_1 	agriculture clear primary water 
2 train_2 	clear primary 
3 train_3 	clear primary 
4 train_4 	agriculture clear habitation primary road
```

Next, we check to see which unique values the labels can take. To do this, and for use later on in the code, we use the data-frame we have made above to create two mapping dictionaries. The first dictionary has class labels as keys and integers as values and the second goes the other way around. The function which does in the code is given below: 

```python
   def get_tag_mapping(mapping_df):
	   ...
	   return(...)
```

Calling this function will result in the following two dictionaries: 

```
{'agriculture': 0, 'artisinal_mine': 1, 'bare_ground': 2, 'blooming': 3, 'blow_down': 4, 'clear': 5, 'cloudy': 6, 'conventional_mine': 7, 'cultivation': 8, 'habitation': 9, 'haze': 10, 'partly_cloudy': 11, 'primary': 12, 'road': 13, 'selective_logging': 14, 'slash_burn': 15, 'water': 16}
```
 ```
 {0: 'agriculture', 1: 'artisinal_mine', 2: 'bare_ground', 3: 'blooming', 4: 'blow_down', 5: 'clear', 6: 'cloudy', 7: 'conventional_mine', 8: 'cultivation', 9: 'habitation', 10: 'haze', 11: 'partly_cloudy', 12: 'primary', 13: 'road', 14: 'selective_logging', 15: 'slash_burn', 16: 'water'}
 ```  

Next, we need to relate these functions to the images. For that, we use a function called: 
```python
   def create_file_mapping(mapping_df):
   	...
	return(...)
```
This function returns a third dictionary with keys as image file names and values as a list of all the labels that are given for that picture. An example of the output for the first training image is given below: 
```
{'train_0': ['haze', 'primary'], 'train_1': ['agriculture', 'clear', 'primary', 'water']}
```

These dictionaries are useful for us not only to know the unique labels in the data-set but also because all supervised machine learning algorithms take only numbers as input and not strings. We need to one-hot encode the information we have. These dictionaries will allow us to do so. The function which does the one-hot encoding for us is given in the code below. A sample function call with sample output is also given : 

```python
   def one_hot_encode(tags, labels_map):
	   ...
	   return(...)
   
   print(one_hot_encode(['agriculture', 'clear', 'primary', 'water'], labels_map))
   
   [1 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 1]
```

How does the function know where to put 1s and 0s? Agriculture is one of the labels. The integer for agriculture according to our previous mapping dictionary is 0. So the 0th element of the list above is given a 1 value by the one-hot encoding. The value of clear is 5 according to our label to integer dictionary. So the value at index 5 of the list above is given a 1, and so on. Now we are ready to compress the data-set and store it.

#### Images 

First we compress the images to 32 X 32 or any other desired target size and then convert these to a numerical array. We need to compress the images if we are running this without a GPU regardless of whether they are on our local machine or on a cloud server. This is to keep the training time manageable as this is just a test case for us to learn the data processing pipeline for satellite imagery. The functions which do this in the code are: 

```python 
   def compress_dataset(path, file_mapping, tag_mapping, target_size = (128, 128)):
	...	
	return(X, y)

   def prep_data(folder = 'train-jpg/', target_size = (32, 32)):
	...
	np.savez_compressed('planet_data.npz', X, y)
	return(...)
```

These two functions will output a file to the repository folder	 with format **.npz** which is a Python built-in compressed file format for numerical arrays. The file will be called **planet_data.npz**. Now that this is done we are finally ready to move ahead with training.

# Model evaluation measure

The model evaluation measure used for this competition is a metric called the F-beta score. Coventionally classifiers use either precision or recall as evaluation metrics. There are many ways to combine these metrics. One common way is to plot the area under the the receiver operating characteristic (ROC) curve which graphs false negatives on the X-axis versus true positives on the Y-axis. This competition uses another combined metric called the F-beta score. This is a weighted average of precision and recall. The entire function is given below. We had to code it manually as Keras does not support this function.

```python
   # Calculate fbeta score for multi-class/label classification
   def fbeta(y_true, y_pred, beta=2):

    	# Clip predictions
    	y_pred = keras.backend.clip(y_pred, 0, 1)
    	
	# Calculate true positives
    	tp = keras.backend.sum(keras.backend.round(keras.backend.clip(y_true * y_pred, 0, 1)), axis=1)
    	
	# Calculate false positives
    	fp = keras.backend.sum(keras.backend.round(keras.backend.clip(y_pred - y_true, 0, 1)), axis=1)
    	
	# Calculate false negatives
    	fn = keras.backend.sum(keras.backend.round(keras.backend.clip(y_true - y_pred, 0, 1)), axis=1)
    	
	# Calculate precision
    	p = tp / (tp + fp + keras.backend.epsilon())
    	
	# Calculate recall
    	r = tp / (tp + fn + keras.backend.epsilon())
    	
	# Calculate fbeta, averaged across each class
    	bb = beta ** 2
    	
	# F-beta score final calculation
    	fbeta_score = keras.backend.mean((1 + bb) * (p * r) / (bb * p + r + keras.backend.epsilon()))
    	
	# Return statement
    	return(fbeta_score)
```


The first set of lines calculates the precision and the recall. ``bb`` is just a weighting function. The final set of lines puts everything together and returns the calculated score. This is tracked throughout training.

# How to prepare hardware

We are using a non-GPU Ubuntu 18.04 Node on E2E Cloud for training. Any other cloud service provider should also work after making small adjustments. Amazon Web Services, Digital Ocean, Microsot Azure, and Google Cloud all provide Ubuntu 18.04 servers. You must have .ssh root access to the machine in order to run the steps above. Steps for the E2E setup are given below: 
	
   * Start a new node or instance
   * Use SSH to log in to your newly created instance as the root user
   * After logging in to your instance run the `setup.sh` file above
   * The `setup.sh` file will install all the software that is needed to run the model onto the machine automatically
   * It will also call the `requirements.txt` file and install all the Python packages into the enviromnent that are needed
   * Once this has run successfully you should be ready to train your model

# Training a base-line model

The model is defined and trained using the `keras` deep learning library. The function which defines the model is called: 

```python
   def define_model(...)
	...
   	return(...)
```

We have written all our code using a Jupyter notebook. Jupyter notebooks are usually designed to be called automatically and do not take in command line arguments out of the box. This makes it hard to experiment with different parameter settings. To overcome this problem we use a Python package called `papermill`. `papermill` is built on top of Jupyter notebook and allows us to call this notebook like a script. We can use the following command to call the notebook: 
	
`papermill planet-rep.ipynb planet-rep-train.ipynb -p epochs 50 -p train True`

This tells `papermill` to run `planet-rep.ipynb` and send all output to `planet-rep-train.ipynb`. It also sets the code to run for 50 epochs and sets the `train` parameter to `True`. When the `train` parameter is set to `True` the file compression parts are skipped. The script directly loads the planet-data.npz file and sends it as an input to the model. 


# Improve Model Performance

  #### Hyper-parameter optimization
  
  #### Data augmentation 
  
  #### More hidden layers
  
  #### Dropout regularization
  
  #### Transfer learning


# Finalise the model and make predictions 


# Comparison to winner's solution 


# Conclusion
