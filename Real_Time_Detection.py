#!/usr/bin/env python
# coding: utf-8

# Uploading Necessary Libraries

# In[1]:


import os
import os.path
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from timeit import default_timer as timer
from time import perf_counter
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score
from IPython.display import Markdown, display
import cv2
import numpy as np
import os
from scipy import stats
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from keras.models import Sequential, model_from_json
from keras.layers import Conv2D, MaxPooling2D, Activation, Dense, Flatten
import joblib
import pickle


# Loading the model from the JSON Files and uploading the weights for Custom model

# In[2]:


# load json and create model
json_file = open('custom_cnn_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("custom_cnn_model.h5")
print("Loaded model from disk")


# Compiling the model

# In[3]:


loaded_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


# Reading the Labels

# In[4]:


with open('labels_detection.pickle', 'rb') as f:
    labels1 = pickle.load(f)


# In[5]:


labels1[28]


# defining the hand area

# In[6]:


def hand_area(img):
    hand = img[100:324, 100:324]
    hand = cv2.resize(hand, (224,224))
    return hand


# Opening the webcam for videocapture

# In[7]:


cap = cv2.VideoCapture(0)
if (cap.isOpened() == False):
    print('Error while trying to open camera. Plese check again...')
# get the frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
# # define codec and create VideoWriter object
out = cv2.VideoWriter('./asl.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (frame_width,frame_height))


# Prediction

# In[ ]:


while(cap.isOpened()):
    # capture each frame of the video
    ret, frame = cap.read()
    # get the hand area on the video capture screen
    cv2.rectangle(frame, (100, 100), (324, 324), (20, 34, 255), 2)
    hand = hand_area(frame)
    image = hand
    image = cv2.flip(image, 1)
#     image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    image = np.expand_dims(image, axis=0)
    pred = loaded_model.predict(image)
    pred = np.argmax(pred,axis=1)
    print(pred)
#     pred = labels1[pred[]]
    pred = (pred[0])
    pred = labels1[pred]
    print(pred)
    cv2.putText(frame, pred, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('image', frame)
    out.write(frame)
    # press `q` to exit
    if cv2.waitKey(27) & 0xFF == ord('q'):  
        break


# In[10]:


cap.release()


# In[11]:


cv2.destroyAllWindows()




