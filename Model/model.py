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

# Insert the file path of the dataset
print("Type the exact file path of the MIMIC-TC v3 Dataset.")
print("Example: G:\Python Projects\Science Research Project")
dataset_file_path = input()

RI_dataset = tf.keras.utils.image_dataset_from_directory(fr'{dataset_file_path}\MIMIC-TC Dataset v3\Training Set',
                                                                 image_size=(300,300),
                                                                 batch_size=32)
print("Loaded RI Dataset!")
print(type(RI_dataset))


# creating the model
model = keras.Sequential()

# adding model layers
model.add(layers.Convolution2D(filters=5, kernel_size=5, stride=1, activation='relu'))
model.add(layers.MaxPooling2D(filters=2, kernel_size=2, stride=2))
model.add(layers.BatchNormalization())
model.add(layers.Convolution2D(filters=5, kernel_size=5, stride=1, activation='relu'))
model.add(layers.MaxPooling2D(filters=5, stride=2))
model.add(layers.BatchNormalization())

model.add(layers.Convolution2D(filters=3, kernel_size=3, stride=1, activation='relu'))
model.add(layers.Convolution2D(filters=3, kernel_size=3, stride=1, activation='relu'))
model.add(layers.Convolution2D(filters=3, kernel_size=3, stride=1, activation='relu'))
model.add(layers.MaxPooling2D(filters=2, stride=2))

model.add(layers.Convolution2D(filters=3, kernel_size=3, stride=1, activation='relu'))
model.add(layers.Convolution2D(filters=3, kernel_size=3, stride=1, activation='relu'))
model.add(layers.Convolution2D(filters=3, kernel_size=3, stride=1, activation='relu'))
model.add(layers.MaxPooling2D(filters=2, stride=2))

model.add(layers.Convolution2D(filters=3, kernel_size=3, stride=1, activation='relu'))
model.add(layers.Convolution2D(filters=3, kernel_size=3, stride=1, activation='relu'))
model.add(layers.Convolution2D(filters=3, kernel_size=3, stride=1, activation='relu'))
model.add(layers.MaxPooling2D(filters=2, stride=2))

model.add(layers.Convolution2D(filters=3, kernel_size=3, stride=1, activation='relu'))
model.add(layers.Convolution2D(filters=3, kernel_size=3, stride=1, activation='relu'))
model.add(layers.Convolution2D(filters=3, kernel_size=3, stride=1, activation='relu'))
model.add(layers.Convolution2D(filters=2, kernel_size=2, stride=2, activation='relu'))

model.add(layers.Dense(activation='relu'))
model.add(layers.Dense(activation='relu'))
model.add(layers.Dense(activation='softmax'))

# compiling model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# training the model
model.fit