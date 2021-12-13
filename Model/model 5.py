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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle

from loading_RI_dataset_3 import X_RI, y_RI
X_RI = pickle.load(open("X_RI.pickle","rb"))
y_RI = pickle.load(open("y_RI.pickle","rb"))

X_RI = X_RI/255.0

# creating the model
model = Sequential()

model.add(layers.Conv2D(64, filters=5, kernel_size=5, activation='relu', input_shape=(300,300,1)))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64, filters=5, kernel_size=5, activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.BatchNormalization())

model.add(layers.Conv2D(64, filters=3, kernel_size=3, activation='relu'))
model.add(layers.Conv2D(64, filters=3, kernel_size=3, activation='relu'))
model.add(layers.Conv2D(64, filters=3, kernel_size=3, activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Conv2D(64, filters=3, kernel_size=3, activation='relu'))
model.add(layers.Conv2D(64, filters=3, kernel_size=3, activation='relu'))
model.add(layers.Conv2D(64, filters=3, kernel_size=3, activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Conv2D(64, filters=3, kernel_size=3, activation='relu'))
model.add(layers.Conv2D(64, filters=3, kernel_size=3, activation='relu'))
model.add(layers.Conv2D(64, filters=3, kernel_size=3, activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Conv2D(64, filters=3, kernel_size=3, activation='relu'))
model.add(layers.Conv2D(64, filters=3, kernel_size=3, activation='relu'))
model.add(layers.Conv2D(64, filters=3, kernel_size=3, activation='relu'))
model.add(layers.Conv2D(64, filters=2, kernel_size=2, activation='relu'))

model.add(layers.Dense(1000, activation='relu'))
model.add(layers.Dense(200, activation='relu'))
model.add(layers.Dense(29, activation='softmax'))

#model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]))
#model.add(Activation("relu"))
#model.add(MaxPooling2D(pool_size=(2,2)))

#model.add(Conv2D(64, (3,3)))
#model.add(Activation("relu"))
#model.add(MaxPooling2D(pool_size=(2,2)))

#model.add(Flatten())
#model.add(Dense(64))

#model.add(Dense(1))
#model.add(Activation("sigmoid"))

# compiling model
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])

X_RI = np.array(X_RI).reshape(-1, 300, 300, 1)
y_RI = np.array(y_RI)

# fitting the model
model.fit(X_RI, y_RI, epochs=10, batch_size=32, validation_split=0.3, callbacks=[tensorboard_callback])

#