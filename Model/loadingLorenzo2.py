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
from sklearn.model_selection import train_test_split
from numpy import asarray

# Insert the file path of the MIMIC-TC v3 Dataset
print("Type the exact file path of the MIMIC-TC v3 Dataset.")
print(fr"Example: G:\Python Projects\Science Research Project")
dataset_file_path = input()

# Insert the TC name
print("Type the TC name.")
print("Examples:")
print("Lorenzo")
print("FIFTEEN - 2019")
print("ALPHA")
storm_name = input()

# Insert Category
print('''Type "Current RI", "Possible RI", or "No RI" without the quotation marks''')
category_name = input()

# Storm Directory filepath
storm_filepath = fr"{dataset_file_path}\MIMIC-TC Dataset v3\Testing Set\{storm_name}\{category_name}"
print(storm_filepath)

# Categories
CATEGORIES = ["Current RI","Possible RI","No RI"]

# loading storm_name directory
storm_files = glob.glob(os.path.join(storm_filepath,"*.png"))

# loading RI Model
model = tf.keras.models.load_model('RI_CNN.model')

def prepare(image):
    img_array = cv2.imread(os.path.join(storm_filepath, image), cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (300, 300))
    return new_array.reshape(-1, 300, 300, 1)

for f in storm_files:
    img_array = cv2.imread(os.path.join(storm_filepath, f), cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (300, 300))
    new_array.reshape(-1, 300, 300, 1)

    # print the location and filename
    print('Location:', f)
    print('File Name:', f.split("\\")[-1])

    # prediction
    prediction = model.predict(prepare(f))
    print(prediction)


