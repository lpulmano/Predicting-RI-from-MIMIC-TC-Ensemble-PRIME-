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

# Training Dataset Processing
# use glob to get the training csv files
path = os.getcwd()
# training csv files
undergoing_csv_files = glob.glob(os.path.join(r"D:\Science Research 2\MIMIC-TC Dataset v3\Training Dataset\Current RI","*.csv"))
likely_csv_files = glob.glob(os.path.join(r"D:\Science Research 2\MIMIC-TC Dataset v3\Training Dataset\Possible RI","*.csv"))
notL_csv_files = glob.glob(os.path.join(r"D:\Science Research 2\MIMIC-TC Dataset v3\Training Dataset\No RI","*.csv"))

# variable for file renaming
i=0

# Currently Undergoing RI processing
for f in undergoing_csv_files:
    # read the csv file
    undergoing_f = pd.read_csv(f, header=None)

    # Converting pandas dataframes to numpy arrays
    undergoing_f = undergoing_f.to_numpy()

    # Converting NaN values to 0
    undergoing_f = np.nan_to_num(undergoing_f, nan=0.0)

    # Converting float array to int array
    undergoing_f = np.around(undergoing_f)
    undergoing_f = undergoing_f.astype(int)

    # converting array to image
    w, h = 300, 300
    undergoing_img = [np.zeros((h, w)).astype(int), np.zeros((h, w)).astype(int), np.zeros((h, w)).astype(int)]
    undergoing_img[0:300] = undergoing_f
    undergoing_img = Image.fromarray(undergoing_f)

    # saving images
    undergoing_path = r'G:\Python Projects\Science Research Project\MIMIC-TC Dataset v3\Training Set\Current RI'
    undergoing_files = os.listdir(undergoing_path)
    i = i+1

    if os.path.exists(undergoing_path+f'\cRI {i}.png'):
        plt.imsave(undergoing_path+fr'\cRI {i+1}.png'.format(int()), undergoing_img)
    else:
        plt.imsave(undergoing_path+fr'\cRI {i}.png', undergoing_img)

# undergoing files are processed
print('''"Currently Undergoing RI" files have been processed''')

# resetting file naming variable
i=0

# Likely to Undergo RI processing
for f in likely_csv_files:
    # read the csv file
    likely_f = pd.read_csv(f, header=None)

    # Converting pandas dataframes to numpy arrays
    likely_f = likely_f.to_numpy()

    # Converting NaN values to 0
    likely_f = np.nan_to_num(likely_f, nan=0.0)

    # Converting float array to int array
    likely_f = np.around(likely_f)
    likely_f = likely_f.astype(int)

    # converting array to image
    w, h = 300, 300
    likely_img = [np.zeros((h, w)).astype(int), np.zeros((h, w)).astype(int), np.zeros((h, w)).astype(int)]
    likely_img[0:300] = likely_f
    likely_img = Image.fromarray(likely_f)

    # saving images
    likely_path = r'G:\Python Projects\Science Research Project\MIMIC-TC Dataset v3\Training Set\Possible RI'
    likely_files = os.listdir(likely_path)
    i = i+1

    if os.path.exists(likely_path+f'\pRI {i}.png'):
        plt.imsave(likely_path+fr'\pRI {i+1}.png'.format(int()), likely_img)
    else:
        plt.imsave(likely_path+fr'\pRI {i}.png', likely_img)

# Likely to Undergo RI files are processed
print('''"Likely to Undergo RI" files have been processed''')

# resetting file naming variable
i=0

# Not Likely to Undergo RI processing
for f in notL_csv_files:
    # read the csv file
    notL_f = pd.read_csv(f, header=None)

    # Converting pandas dataframes to numpy arrays
    notL_f = notL_f.to_numpy()

    # Converting NaN values to 0
    notL_f = np.nan_to_num(notL_f, nan=0.0)

    # Converting float array to int array
    notL_f = np.around(notL_f)
    notL_f = notL_f.astype(int)

    # converting array to image
    w, h = 300, 300
    notL_img = [np.zeros((h, w)).astype(int), np.zeros((h, w)).astype(int), np.zeros((h, w)).astype(int)]
    notL_img[0:300] = notL_f
    notL_img = Image.fromarray(notL_f)

    # saving images
    notL_path = r"G:\Python Projects\Science Research Project\MIMIC-TC Dataset v3\Training Set\No RI"
    notL_files = os.listdir(notL_path)
    i = i+1

    if os.path.exists(notL_path+fr"\nRI {i}.png"):
        plt.imsave(notL_path+fr'\nRI {i+1}.png'.format(int()), notL_img)
    else:
        plt.imsave(notL_path+fr'\nRI {i}.png', notL_img)

# Not Likely to Undergo RI files are processed
print('''"Not Likely to Undergo RI" files have been processed''')

# resetting file naming variable
i=0

# Loading train images into keras
RI_training_dataset = tf.keras.preprocessing.image_dataset_from_directory('G:\Python Projects\Science Research Project\MIMIC-TC Dataset v3\Training Set', image_size=(300,300), batch_size=32)
print(type(RI_training_dataset))
print("Loaded Training Dataset")



# Testing Dataset Processing

# ALPHA csv processing
test_csv_files = glob.glob(os.path.join(r"D:\Science Research 2\MIMIC-TC Dataset v3\Testing Dataset\ALPHA\Current RI","*.csv"))
for f in test_csv_files:
    # read the csv file
    test_f = pd.read_csv(f, header=None)

    # Converting pandas dataframes to numpy arrays
    test_f = test_f.to_numpy()

    # Converting NaN values to 0
    test_f = np.nan_to_num(test_f, nan=0.0)

    # Converting float array to int array
    test_f = np.around(test_f)
    test_f = test_f.astype(int)

    # converting array to image
    w, h = 300, 300
    test_img = [np.zeros((h, w)).astype(int), np.zeros((h, w)).astype(int), np.zeros((h, w)).astype(int)]
    test_img[0:300] = test_f
    test_img = Image.fromarray(test_f)

    # saving images
    ALPHA_path = r"G:\Python Projects\Science Research Project\MIMIC-TC Dataset v3\Testing Set\ALPHA\Current RI"
    ALPHA_files = os.listdir(ALPHA_path)
    i = i+1

    if os.path.exists(ALPHA_path+fr"\ALPHA cRI {i}.png"):
        plt.imsave(ALPHA_path+fr'\ALPHA cRI {i+1}.png'.format(int()), test_img)
    else:
        plt.imsave(ALPHA_path+fr'\ALPHA cRI {i}.png', test_img)

# resetting file naming variable
i=0

test_csv_files = glob.glob(os.path.join(r"D:\Science Research 2\MIMIC-TC Dataset v3\Testing Dataset\ALPHA\Possible RI","*.csv"))
for f in test_csv_files:
    # read the csv file
    test_f = pd.read_csv(f, header=None)

    # Converting pandas dataframes to numpy arrays
    test_f = test_f.to_numpy()

    # Converting NaN values to 0
    test_f = np.nan_to_num(test_f, nan=0.0)

    # Converting float array to int array
    test_f = np.around(test_f)
    test_f = test_f.astype(int)

    # converting array to image
    w, h = 300, 300
    test_img = [np.zeros((h, w)).astype(int), np.zeros((h, w)).astype(int), np.zeros((h, w)).astype(int)]
    test_img[0:300] = test_f
    test_img = Image.fromarray(test_f)

    # saving images
    ALPHA_path = r"G:\Python Projects\Science Research Project\MIMIC-TC Dataset v3\Testing Set\ALPHA\Possible RI"
    ALPHA_files = os.listdir(ALPHA_path)
    i = i+1

    if os.path.exists(ALPHA_path+fr"\ALPHA pRI {i}.png"):
        plt.imsave(ALPHA_path+fr'\ALPHA pRI {i+1}.png'.format(int()), test_img)
    else:
        plt.imsave(ALPHA_path+fr'\ALPHA pRI {i}.png', test_img)

# resetting file naming variable
i=0

test_csv_files = glob.glob(os.path.join(r"D:\Science Research 2\MIMIC-TC Dataset v3\Testing Dataset\ALPHA\No RI","*.csv"))
for f in test_csv_files:
    # read the csv file
    test_f = pd.read_csv(f, header=None)

    # Converting pandas dataframes to numpy arrays
    test_f = test_f.to_numpy()

    # Converting NaN values to 0
    test_f = np.nan_to_num(test_f, nan=0.0)

    # Converting float array to int array
    test_f = np.around(test_f)
    test_f = test_f.astype(int)

    # converting array to image
    w, h = 300, 300
    test_img = [np.zeros((h, w)).astype(int), np.zeros((h, w)).astype(int), np.zeros((h, w)).astype(int)]
    test_img[0:300] = test_f
    test_img = Image.fromarray(test_f)

    # saving images
    ALPHA_path = r"G:\Python Projects\Science Research Project\MIMIC-TC Dataset v3\Testing Set\ALPHA\No RI"
    ALPHA_files = os.listdir(ALPHA_path)
    i = i+1

    if os.path.exists(ALPHA_path+f"\ALPHA nRI {i}.png"):
        plt.imsave(ALPHA_path+fr'\ALPHA nRI {i+1}.png'.format(int()), test_img)
    else:
        plt.imsave(ALPHA_path+fr'\ALPHA nRI {i}.png', test_img)

# resetting file naming variable
i=0

# ALPHA files are processed
print('''ALPHA files have been processed''')

# Loading test datasets into keras
ALPHA_dataset = tf.keras.preprocessing.image_dataset_from_directory('G:\Python Projects\Science Research Project\MIMIC-TC Dataset v3\Testing Set\ALPHA', image_size=(300,300))
print("Loaded ALPHA Dataset")
