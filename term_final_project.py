# -*- coding: utf-8 -*-
"""Tiwari_Gaurav_METCS767_Term_Final_Project.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/11ZgY_JOsPEkkca6bcoUlH8ylSHoLIGV7
"""

## Installing the mediapipe library
!pip install mediapipe

## Importing necessary libraries
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
import pickle

## defining markdown function
def md(string):   
    display(Markdown(string))

from google.colab import drive
drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
# %cd drive/MyDrive

"""Data preprocessing and visualization"""

# Create a list with the filepaths for training and testing
dir_ = Path('/content/drive/MyDrive/asl_alphabet_train')
file_paths = list(dir_.glob(r'**/*.jpg'))

def process_img(filepath):
    """ Create a DataFrame with the filepath and the labels of the pictures
    """

    labels = [str(filepath[i]).split("/")[-2] \
              for i in range(len(filepath))]

    filepath = pd.Series(filepath, name='Filepath').astype(str)
    labels = pd.Series(labels, name='Label')

    # Concatenate filepaths and labels
    df = pd.concat([filepath, labels], axis=1)

    # Shuffle the DataFrame and reset index
    df = df.sample(frac=1,random_state=0).reset_index(drop = True)
    
    return df

df = process_img(file_paths)

print(f'Number of pictures in the dataset: {df.shape[0]}\n')
print(f'Number of different labels: {len(df.Label.unique())}\n')
print(f'Labels: {df.Label.unique()}')

# The DataFrame with the filepaths in one column and the labels in the other one
df.head(5)

# Display the number of pictures of each category
counts = df['Label'].value_counts()
plt.figure(figsize=(20,5))
sns.barplot(x = sorted(counts.index), y = counts, palette = "crest")
plt.title("Number of pictures of each category", fontsize = 15)
plt.show()

# Displaying the shape of each image
k = plt.imread(df.Filepath[1])
k.shape

### plotting few samples
fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(15, 7),
                        subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(df.Filepath[i]))
    ax.set_title(df.Label[i], fontsize = 15)
plt.tight_layout(pad=0.5)
plt.show()

# Split into training and test datasets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=0)

len(train_df)

"""Load the Images with a generator and Data Augmentation"""

# Image Augmentation and Image generator
def create_generator():
    # Load the Images with a generator and Data Augmentation
    train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,validation_split=0.25
    )


    test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
    )

    train_images = train_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=True,
        seed=0,
        subset='training',
                rotation_range=30, # Uncomment to use data augmentation
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=False,
        fill_mode="nearest",
    )

    val_images = train_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=False,
        seed=0,
        subset='validation',
        rotation_range=30, # Uncomment to use data augmentation
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    test_images = test_generator.flow_from_dataframe(
        dataframe=test_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=False
    )
    
    return train_generator,test_generator,train_images,val_images,test_images

"""Load the generator without data augmentation"""

def create_generator1():
    # Load the Images with a generator and Data Augmentation
    train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=None,validation_split=0.25
    )


    test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=None
    )

    train_images = train_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=True,
        seed=0,
        subset='training',
    )

    val_images = train_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=False,
        seed=0,
        subset='validation',
    )
    test_images = test_generator.flow_from_dataframe(
        dataframe=test_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=False
    )
    
    return train_generator,test_generator,train_images,val_images,test_images

## Doing transfer learning from few models and building the model out of it

def get_model(model):
# Load the pretained model
    kwargs =    {'input_shape':(224, 224, 3),
                'include_top':False,
                'weights':'imagenet',
                'pooling':'avg'}
    
    pretrained_model = model(**kwargs)
    pretrained_model.trainable = False
    
    inputs = pretrained_model.input

    x = tf.keras.layers.Dense(128, activation='relu')(pretrained_model.output)
    x = tf.keras.layers.Dense(128, activation='relu')(x)

    outputs = tf.keras.layers.Dense(29, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

## Making  custom CNN model out of it
def get_created_model():

  model1 = Sequential()

  model1.add(Conv2D(32, (5, 5), input_shape=(224, 224, 3)))
  model1.add(Activation('relu'))
  model1.add(MaxPooling2D((2, 2)))

  model1.add(Conv2D(64, (3, 3)))
  model1.add(Activation('relu'))
  model1.add(MaxPooling2D((2, 2)))

  model1.add(Conv2D(64, (3, 3)))
  model1.add(Activation('relu'))
  model1.add(MaxPooling2D((2, 2)))

  model1.add(Flatten())

  model1.add(Dense(128, activation='relu'))

  model1.add(Dense(29, activation='softmax'))

  model1.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

  return model1

## declaring Pre-trained models for transfer learning
pre_trained_models = {
    "DenseNet121": {"model":tf.keras.applications.DenseNet121, "perf":0},
    "MobileNetV2": {"model":tf.keras.applications.MobileNetV2, "perf":0},
    "DenseNet169": {"model":tf.keras.applications.DenseNet169, "perf":0},
    "DenseNet201": {"model":tf.keras.applications.DenseNet201, "perf":0},
    "EfficientNetB0": {"model":tf.keras.applications.EfficientNetB0, "perf":0},
    "MobileNet": {"model":tf.keras.applications.MobileNet, "perf":0},
    "ResNet152": {"model":tf.keras.applications.ResNet152, "perf":0},
    "ResNet50": {"model":tf.keras.applications.ResNet50, "perf":0},
    "VGG16": {"model":tf.keras.applications.VGG16, "perf":0},
    "VGG19": {"model":tf.keras.applications.VGG19, "perf":0},
    "Xception": {"model":tf.keras.applications.Xception, "perf":0}
}

# Calling train generator for the Image Augmentation and calling different pre-trained models to find out the best model for training.
train_generator,test_generator,train_images,val_images,test_images=create_generator()
print('\n')

# Fit the models
for name, model in pre_trained_models.items():
    
    # Get the model
    mod = get_model(model['model'])
    pre_trained_models[name]['model'] = mod
    
    start = perf_counter()
    
    # Fit the model
    history = mod.fit(train_images,validation_data=val_images,epochs=1,verbose=0)
    
    # Sav the duration and the val_accuracy
    duration = perf_counter() - start
    duration = round(duration,2)
    pre_trained_models[name]['perf'] = duration
    print(f"{name:20} trained in {duration} sec")
    
    val_acc = history.history['val_accuracy']
    pre_trained_models[name]['val_acc'] = [round(v,4) for v in val_acc]

# Create a DataFrame with the results
models_result = []

for name, v in pre_trained_models.items():
    models_result.append([ name, pre_trained_models[name]['val_acc'][-1], 
                          pre_trained_models[name]['perf']])
    
df_results = pd.DataFrame(models_result, 
                          columns = ['model','val_accuracy','Training time (sec)'])
df_results.sort_values(by='val_accuracy', ascending=False, inplace=True)
df_results.reset_index(inplace=True,drop=True)
df_results

## Plotting the figure of the accuracy after one epoch for each transfer learning models
plt.figure(figsize = (15,5))
sns.barplot(x = 'model', y = 'val_accuracy', data = df_results)
plt.title('Accuracy on the test set (after 1 epoch))', fontsize = 15)
plt.ylim(0,1)
plt.xticks(rotation=90)
plt.show()

## Plotting the training time
plt.figure(figsize = (15,5))
sns.barplot(x = 'model', y = 'Training time (sec)', data = df_results)
plt.title('Training time for each model in sec', fontsize = 15)
plt.xticks(rotation=90)
plt.show()

## fitting the model on the generated images
# Use the whole data which is split into training and test datasets
train_df, test_df = train_test_split(df, test_size=0.1, random_state=0)

# Create the generators
train_generator,test_generator,train_images,val_images,test_images=create_generator()

train_generator1,test_generator1,train_images1,val_images1,test_images1=create_generator1()

# Get the model with the highest validation score
best_model = df_results.iloc[0]
model = get_model( eval("tf.keras.applications."+ best_model[0]) )
model2 = get_model( eval("tf.keras.applications."+ best_model[0]) )
# Create a new model
model1 = get_created_model()
model3 = get_created_model()
# Train the model
history = model.fit(train_images,validation_data=val_images,epochs=5,callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=1,restore_best_weights=True)])

history1 = model2.fit(train_images1,validation_data=val_images1,epochs=5,callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=1,restore_best_weights=True)])

## Custom CNN model fit
history2 = model1.fit(train_images,validation_data=val_images,epochs=5,callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=1,restore_best_weights=True)])

history3 = model3.fit(train_images1,validation_data=val_images1,epochs=5,callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=1,restore_best_weights=True)])

## Defining afunction to print plot for depecting val accuracy and accuracy, loss and val-loss
def le_curve(history,c):
    pd.DataFrame(history.history).plot(figsize=(8, 5)) 
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.title(c, fontsize = 15)
    plt.show()

le_curve(history, "Plot for the Pre-trained model with Data Augmentation")

le_curve(history1, "Plot for the Pre-trained model without Data Augmentation")

le_curve(history2, "Plot for the custom model with Data Augmentation")

le_curve(history3,'Plot for the custom model without Data Augmentation')

fig1, axes1 = plt.subplots(2, 1, figsize=(15, 10))
ax1 = axes1.flat

pd.DataFrame(history.history)[['accuracy','val_accuracy']].plot(ax=ax1[0])
ax1[0].set_title("Accuracy for the transferred learned model with Data Augmentation", fontsize = 15)
ax1[0].set_ylim(0,1.1)

pd.DataFrame(history.history)[['loss','val_loss']].plot(ax=ax1[1])
ax1[1].set_title("Loss for the transferred learned model with Data Augmentation", fontsize = 15)
plt.show()

fig2, axes2 = plt.subplots(2, 1, figsize=(15, 10))
ax2 = axes2.flat

pd.DataFrame(history1.history)[['accuracy','val_accuracy']].plot(ax=ax2[0])
ax2[0].set_title("Accuracy for transfer learning model without Data Augmentation", fontsize = 15)
ax2[0].set_ylim(0,1.1)

pd.DataFrame(history1.history)[['loss','val_loss']].plot(ax=ax2[1])
ax2[1].set_title("Loss for transfer learning model without Data Augmentation", fontsize = 15)
plt.show()

fig3, axes3 = plt.subplots(2, 1, figsize=(15, 10))
ax3 = axes3.flat

pd.DataFrame(history2.history)[['accuracy','val_accuracy']].plot(ax=ax3[0])
ax3[0].set_title("Accuracy for the custom model with Data Augmentation", fontsize = 15)
ax3[0].set_ylim(0,1.1)

pd.DataFrame(history2.history)[['loss','val_loss']].plot(ax=ax3[1])
ax3[1].set_title("Loss for the custom model with Data Augmentation", fontsize = 15)
plt.show()

fig4, axes4 = plt.subplots(2, 1, figsize=(15, 10))
ax4 = axes4.flat

pd.DataFrame(history.history)[['accuracy','val_accuracy']].plot(ax=ax4[0])
ax4[0].set_title("Accuracy for the custom model without Data Augmentation", fontsize = 15)
ax4[0].set_ylim(0,1.1)

pd.DataFrame(history.history)[['loss','val_loss']].plot(ax=ax4[1])
ax4[1].set_title("Loss for the custom model without Data Augmentation", fontsize = 15)
plt.show()

"""Train the architecture with the best result"""

## finding the accuracy on the test set and printing the conf. matrix for the best model out of pre-trained models
# Predict the label of the test_images
pred = model.predict(test_images)
pred = np.argmax(pred,axis=1)

# Map the label
labels = (train_images.class_indices)
labels = dict((v,k) for k,v in labels.items())
pred = [labels[k] for k in pred]

# Get the accuracy on the test set
y_test = list(test_df.Label)
acc = accuracy_score(y_test,pred)

# Display the results
md(f'## Model with data augmentation with {acc*100:.2f}% accuracy on the test set')

# Display a confusion matrix
cf_matrix = confusion_matrix(y_test, pred, normalize='true')
plt.figure(figsize = (17,12))
sns.heatmap(cf_matrix, annot=True, xticklabels = sorted(set(y_test)), yticklabels = sorted(set(y_test)),cbar=False)
plt.title('Normalized Confusion Matrix', fontsize = 23)
plt.xticks(fontsize=12,rotation=45)
plt.yticks(fontsize=12)
plt.show()

## finding the accuracy on the test set and printing the conf. matrix for the best model out of pre-trained models
# Predict the label of the test_images
pred2 = model2.predict(test_images1)
pred2 = np.argmax(pred2,axis=1)

# Map the label
labels2 = (train_images1.class_indices)
labels2 = dict((v,k) for k,v in labels2.items())
pred2 = [labels2[k] for k in pred2]

# Get the accuracy on the test set
y_test = list(test_df.Label)
acc2 = accuracy_score(y_test,pred2)

# Display the results
md(f'## Model without data augmentation with {acc2*100:.2f}% accuracy on the test set')

# Display a confusion matrix
cf_matrix = confusion_matrix(y_test, pred2, normalize='true')
plt.figure(figsize = (17,12))
sns.heatmap(cf_matrix, annot=True, xticklabels = sorted(set(y_test)), yticklabels = sorted(set(y_test)),cbar=False)
plt.title('Normalized Confusion Matrix', fontsize = 23)
plt.xticks(fontsize=12,rotation=45)
plt.yticks(fontsize=12)
plt.show()

model.summary()

train_generator,test_generator,train_images,val_images,test_images=create_generator()

labels = (train_images.class_indices)
labels = dict((v,k) for k,v in labels.items())
labels_detection = dict((k,v) for k,v in labels.items())

labels_detection

with open('labels_detection.pickle', 'wb') as f:
    pickle.dump(labels_detection, f, protocol=pickle.HIGHEST_PROTOCOL)

with open('labels_detection.pickle', 'rb') as f:
    labels1 = pickle.load(f)

labels1[0]

## finding the accuracy on the test set and printing the conf. matrix for the CNN model
# Predict the label of the test_images
pred1 = model1.predict(test_images)
pred1 = np.argmax(pred1,axis=1)

# Map the label
labels = (train_images.class_indices)
labels = dict((v,k) for k,v in labels.items())
pred1 = [labels[k] for k in pred1]

# Get the accuracy on the test set
y_test = list(test_df.Label)
acc1 = accuracy_score(y_test,pred1)

# Display the results
md(f'## Designed Model with data augmentation with {acc1*100:.2f}% accuracy on the test set')

# Display a confusion matrix
cf_matrix = confusion_matrix(y_test, pred1, normalize='true')
plt.figure(figsize = (17,12))
sns.heatmap(cf_matrix, annot=True, xticklabels = sorted(set(y_test)), yticklabels = sorted(set(y_test)),cbar=False)
plt.title('Normalized Confusion Matrix', fontsize = 23)
plt.xticks(fontsize=12,rotation=45)
plt.yticks(fontsize=12)
plt.show()

## finding the accuracy on the test set and printing the conf. matrix for the CNN model
# Predict the label of the test_images
pred3 = model3.predict(test_images1)
pred3 = np.argmax(pred3,axis=1)

# Map the label
labels3 = (train_images1.class_indices)
labels3 = dict((v,k) for k,v in labels3.items())
pred3 = [labels3[k] for k in pred3]

# Get the accuracy on the test set
y_test = list(test_df.Label)
acc3 = accuracy_score(y_test,pred3)

# Display the results
md(f'## Designed Model without data augmentation with {acc3*100:.2f}% accuracy on the test set')

# Display a confusion matrix
cf_matrix = confusion_matrix(y_test, pred3, normalize='true')
plt.figure(figsize = (17,12))
sns.heatmap(cf_matrix, annot=True, xticklabels = sorted(set(y_test)), yticklabels = sorted(set(y_test)),cbar=False)
plt.title('Normalized Confusion Matrix', fontsize = 23)
plt.xticks(fontsize=12,rotation=45)
plt.yticks(fontsize=12)
plt.show()

"""Examples of Prediction"""

model1.summary()

# Display picture of the dataset with their labels
fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(20, 12),
                        subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(test_df.Filepath.iloc[i]))
    ax.set_title(f"True: {test_df.Label.iloc[i].split('_')[0]}\nPredicted: {pred[i].split('_')[0]}", fontsize = 15)
plt.tight_layout()
plt.show()

# Display picture of the dataset with their labels
fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(20, 12),
                        subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(test_df.Filepath.iloc[i]))
    ax.set_title(f"True: {test_df.Label.iloc[i].split('_')[0]}\nPredicted: {pred1[i].split('_')[0]}", fontsize = 15)
plt.tight_layout()
plt.show()

## saving the model to the json file
model1_json = model1.to_json()

with open("/content/drive/MyDrive/custom_cnn_model.json", "w") as json_file:
    json_file.write(model1_json)

## saving the transfer learned model to the json file
model_json = model.to_json()

with open("/content/drive/MyDrive/transfer_model.json", "w") as json_file:
    json_file.write(model_json)

## saving the weights for the model
model1.save_weights("/content/drive/MyDrive/custom_cnn_model.h5")

## saving the weights for transfer learned model
model1.save_weights("/content/drive/MyDrive/transfer_model.h5")

# load json and create model
json_file = open('/content/drive/MyDrive/custom_cnn_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("/content/drive/MyDrive/custom_cnn_model.h5")
print("Loaded model from disk")

loaded_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
score = loaded_model.evaluate(test_images)