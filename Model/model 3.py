import keras
import pandas as pd
import numpy as np
import os
import glob
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
from PIL import Image
import time
import cv2
import seaborn as sns
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.utils import np_utils
import pathlib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.preprocessing.image import ImageDataGenerator
import splitfolders

# Insert the file path of the dataset
print("Type the exact file path of the MIMIC-TC v3 Dataset.")
print("Example: G:\Python Projects\Science Research Project")
dataset_file_path = input()

RI_dataset = tf.keras.utils.image_dataset_from_directory(fr'{dataset_file_path}\MIMIC-TC Dataset v3\Training Set', image_size=(300,300))
print("Loaded RI Dataset!")
print(type(RI_dataset))

# splitting RI_dataset into training, validation, and testing using ratio
input_folder = fr"{dataset_file_path}\MIMIC-TC Dataset v3\Training Set"

splitfolders.ratio(input_folder, output=fr"{dataset_file_path}\MIMIC-TC Dataset v3 - Split\Training Set",
                   seed=10, ratio=(.7, .3))

# splitting RI_dataset into training, validation, and testing using fixed
# splitfolders.fixed(input_folder, output=fr"{dataset_file_path}\MIMIC-TC Dataset v3 - Split\Training Set",
                  # seed=10, fixed=(100,100))

# declaring training and test
X_train = tf.keras.utils.image_dataset_from_directory(fr'{dataset_file_path}\MIMIC-TC Dataset v3 - Split\Training Set', image_size=(300,300))

X_test = tf.keras.utils.image_dataset_from_directory(fr'{dataset_file_path}\MIMIC-TC Dataset v3\Testing Set\Lorenzo', image_size=(300,300))

y_train = tf.keras.utils.image_dataset_from_directory(fr'{dataset_file_path}\MIMIC-TC Dataset v3 - Split\Training Set', image_size=(300,300))

y_test = tf.keras.utils.image_dataset_from_directory(fr'{dataset_file_path}\MIMIC-TC Dataset v3\Testing Set\Lorenzo', image_size=(300,300))

# creating the model
model = keras.Sequential()

# adding model layers
model.add(layers.Convolution2D(filters=5, kernel_size=5, activation='relu'))
model.add(layers.MaxPooling2D())
model.add(layers.BatchNormalization())
model.add(layers.Convolution2D(filters=5, kernel_size=5, activation='relu'))
model.add(layers.MaxPooling2D())
model.add(layers.BatchNormalization())

model.add(layers.Convolution2D(filters=3, kernel_size=3, activation='relu'))
model.add(layers.Convolution2D(filters=3, kernel_size=3, activation='relu'))
model.add(layers.Convolution2D(filters=3, kernel_size=3, activation='relu'))
model.add(layers.MaxPooling2D())

model.add(layers.Convolution2D(filters=3, kernel_size=3, activation='relu'))
model.add(layers.Convolution2D(filters=3, kernel_size=3, activation='relu'))
model.add(layers.Convolution2D(filters=3, kernel_size=3, activation='relu'))
model.add(layers.MaxPooling2D())

model.add(layers.Convolution2D(filters=3, kernel_size=3, activation='relu'))
model.add(layers.Convolution2D(filters=3, kernel_size=3, activation='relu'))
model.add(layers.Convolution2D(filters=3, kernel_size=3, activation='relu'))
model.add(layers.MaxPooling2D())

model.add(layers.Convolution2D(filters=3, kernel_size=3, activation='relu'))
model.add(layers.Convolution2D(filters=3, kernel_size=3, activation='relu'))
model.add(layers.Convolution2D(filters=3, kernel_size=3, activation='relu'))
model.add(layers.Convolution2D(filters=2, kernel_size=2, activation='relu'))

model.add(layers.Dense(1000, activation='relu'))
model.add(layers.Dense(200, activation='relu'))
model.add(layers.Dense(29, activation='softmax'))



# compiling the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# training the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)