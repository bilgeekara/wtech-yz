from fastapi import FastAPI

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt
import time
from os import listdir
import keras


#importing libaries
import numpy as np 
import pandas as pd
import random as rd
import os

import matplotlib.pyplot as plt

import seaborn as sns

from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.metrics import Accuracy
from keras.applications.vgg16 import VGG16


#setting seed for reproducability
from numpy.random import seed
seed(25)
tf.random.set_seed(50)
# Define the FastAPI app
app = FastAPI()
def crop_brain_contour(image, plot=False):
    
    #import imutils
    #import cv2
    #from matplotlib import pyplot as plt
    
    # Convert the image to grayscale, and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold the image, then perform a series of erosions +
    # dilations to remove any small regions of noise
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours in thresholded image, then grab the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    

    # Find the extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    
    # crop new image out of the original image using the four extreme points (left, right, top, bottom)
    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]            

    if plot:
        plt.figure()

        plt.subplot(1, 2, 1)
        plt.imshow(image)
        
        plt.tick_params(axis='both', which='both', 
                        top=False, bottom=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        
        plt.title('Original Image')
            
        plt.subplot(1, 2, 2)
        plt.imshow(new_image)

        plt.tick_params(axis='both', which='both', 
                        top=False, bottom=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        plt.title('Cropped Image')
        
        plt.show()
    
    return new_image
def load_data(dir_list, image_size):
    """
    Read images, resize and normalize them. 
    Arguments:
        dir_list: list of strings representing file directories.
    Returns:
        X: A numpy array with shape = (#_examples, image_width, image_height, #_channels)
        y: A numpy array with shape = (#_examples, 1)
    """

    # load all images in a directory
    X = []
    y = []
    image_width, image_height = image_size
    
    for directory in dir_list:
        for filename in listdir(directory):
            # load the image
            image = cv2.imread(directory + '\\' + filename)
            # crop the brain and ignore the unnecessary rest part of the image
            image = crop_brain_contour(image, plot=False)
            # resize image
            image = cv2.resize(image, dsize=(image_width, image_height), interpolation=cv2.INTER_CUBIC)
            # normalize values
            image = image / 255.
            # convert image to numpy array and append it to X
            X.append(image)
            # append a value of 1 to the target array if the image
            # is in the folder named 'yes', otherwise append 0.
            if directory[-3:] == 'yes':
                y.append([1])
            else:
                y.append([0])
                
    X = np.array(X)
    y = np.array(y)
    
    # Shuffle the data
    X, y = shuffle(X, y)
    
    print(f'Number of examples is: {len(X)}')
    print(f'X shape is: {X.shape}')
    print(f'y shape is: {y.shape}')
    
    return X, y

@app.post("/test/") 
def test():
    return "Test başarılı olarak gerçekleştirildi."


@app.post("/predict_CNN/") # url CNN
def predict_CNN(test_sample_no:int):

    
    best_model = load_model(filepath='models/best_cnn_model08-0.84.keras')
    data_path = 'images/'
    
    # augmented data (yes and no) contains both the original and the new generated examples
    augmented_yes = data_path + 'yes' 
    augmented_no = data_path + 'no'
    
    IMG_WIDTH, IMG_HEIGHT = (240, 240)
    
    X, y = load_data([augmented_yes, augmented_no], (IMG_WIDTH, IMG_HEIGHT))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7)
    y_test_prob = best_model.predict(X_test[:test_sample_no])
    print(y_test_prob)
    print(y_test[:test_sample_no])
    return {"Prediction": str(abs(y_test_prob[0][0])),
            "Real Value" :str(y_test[:test_sample_no][0][0])}


@app.post("/predict_VGG16/") # url CNN
def predict_VGG16(test_sample_no:int):
    
        
    # 0 - Normal
    # 1 - Tumor
    
    data = [] #creating a list for images
    paths = [] #creating a list for paths
    labels = [] #creating a list to put our 0 or 1 labels
    
    #staring with the images that have tumors
    for r, d, f in os.walk(r'./images/yes'):
        for file in f:
            if '.jpg' in file:
                paths.append(os.path.join(r, file))
    
    for path in paths:
        img = Image.open(path)
        img = img.resize((128,128))
        img = np.array(img)
        if(img.shape == (128,128,3)):
            data.append(np.array(img))
            labels.append(1)
    
    #now working with the images with no tumors        
    paths = []
    for r, d, f in os.walk(r"./images/no"):
        for file in f:
            if '.jpg' in file:
                paths.append(os.path.join(r, file))
    
    for path in paths:
        img = Image.open(path)
        img = img.resize((128,128))
        img = np.array(img)
        if(img.shape == (128,128,3)):
            data.append(np.array(img))
            labels.append(0)
            
    data = np.array(data)
    data.shape
    
    labels = np.array(labels)
    labels = labels.reshape(20,1)
    
    print('data shape is:', data.shape)
    print('labels shape is:', labels.shape)
    #reducing the data to between 1 and 0
    data = data / 255.00
    #getting the max of the array
    print(np.max(data))
    #getting the min of the array
    print(np.min(data))
    vgg16_model = load_model("./models/VGG16_models.h5")
    pred=vgg16_model.predict(data[0])
    return {"Prediction": str(pred),
            "Real Value" :str(labels[test_sample_no])}


@app.get("/")
async def read_root():
    return {"message": "Welcome to the Driver State Prediction API. Created by Ömer Faruk Ballı"}
    


# Run the FastAPI app
# python -m uvicorn ann:app --reload
