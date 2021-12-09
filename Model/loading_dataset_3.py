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

# Insert the file path of the training dataset
print("Type the exact file path of the MIMIC-TC v3 Dataset.")
print("Example: G:\Python Projects\Science Research Project")
dataset_file_path = input()
dataset_file_path = fr"{dataset_file_path}\MIMIC-TC Dataset v3\Training Set"

# Categories
CATEGORIES = ["Current RI","Possible RI","No RI"]

# creating training data
training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(dataset_file_path, category)  # path to RI classifications
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (300,300))
            training_data.append([new_array, class_num])

create_training_data()
print(len(training_data))

# shuffling data
import random
random.shuffle(training_data)

for sample in training_data[:10]:
    print(sample[1])

# features and labels
X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

# reshaping
X = np.array(X).reshape(-1, 300, 300, 1)

import pickle

pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

# for category in CATEGORIES:
#     path = os.path.join(dataset_file_path, category)  # path to RI classifications
#     for img in os.listdir(path):
#         img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
#         plt.imshow(img_array, cmap="gray")
#         plt.show()
#         break
#     break


# IMG_SIZE = 300
#
# new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
# plt.imshow(new_array, cmap='gray')