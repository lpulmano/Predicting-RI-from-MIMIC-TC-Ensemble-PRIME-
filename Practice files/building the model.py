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

# use glob to get the training csv files
path = os.getcwd()
undergoing_csv_files = glob.glob(os.path.join(r"D:\Science Research 2\MIMIC-TC Dataset v2\Training Dataset\Currently Undergoing RI","*.csv"))
likely_csv_files = glob.glob(os.path.join(r"D:\Science Research 2\MIMIC-TC Dataset v2\Training Dataset\Likely to Undergo RI","*.csv"))
notL_csv_files = glob.glob(os.path.join(r"D:\Science Research 2\MIMIC-TC Dataset v2\Training Dataset\Likely Not to Undergo RI","*.csv"))

# variable for file renaming
i=0

# Currently Undergoing RI
for f in undergoing_csv_files:
    # read the csv file
    undergoing_df = pd.read_csv(f, header=None)

    # Converting pandas dataframes to numpy arrays
    undergoing_arr = undergoing_df.to_numpy()

    # Converting NaN values to 0
    undergoing_farray = np.nan_to_num(undergoing_arr, nan=0.0)

    # Converting float array to int array
    undergoing_farray = np.around(undergoing_farray)
    undergoing_iarray = undergoing_farray.astype(int)

    # converting array to image
    w, h = 300, 300
    undergoing_img = [np.zeros((h, w)).astype(int), np.zeros((h, w)).astype(int), np.zeros((h, w)).astype(int)]
    undergoing_img[0:300] = undergoing_iarray
    undergoing_img = Image.fromarray(undergoing_iarray)

    # saving images
    undergoing_path = 'G:\Python Projects\Science Research Project\MIMIC-TC Dataset v3\Training Set\Currently Undergoing RI'
    undergoing_files = os.listdir(undergoing_path)
    i = i+1

    if os.path.exists(undergoing_path+f'\Currently {i}.png'):
        plt.imsave(undergoing_path+f'\Currently {i+1}.png'.format(int()), undergoing_img)
    else:
        plt.imsave(undergoing_path+f'\Currently {i}.png', undergoing_img)

# undergoing files are processed
print('''"Currently Undergoing RI" files have been processed''')

# resetting file naming variable
i=0

# Likely to Undergo RI
for f in likely_csv_files:
    # read the csv file
    likely_df = pd.read_csv(f, header=None)

    # Converting pandas dataframes to numpy arrays
    likely_arr = likely_df.to_numpy()

    # Converting NaN values to 0
    likely_farray = np.nan_to_num(likely_arr, nan=0.0)

    # Converting float array to int array
    likely_farray = np.around(likely_farray)
    likely_iarray = likely_farray.astype(int)

    # converting array to image
    w, h = 300, 300
    likely_img = [np.zeros((h, w)).astype(int), np.zeros((h, w)).astype(int), np.zeros((h, w)).astype(int)]
    likely_img[0:300] = likely_iarray
    likely_img = Image.fromarray(likely_iarray)

    # saving images
    likely_path = 'G:\Python Projects\Science Research Project\MIMIC-TC Dataset v3\Training Set\Likely to Undergo RI'
    likely_files = os.listdir(likely_path)
    i = i+1

    if os.path.exists(likely_path+f'\Likely {i}.png'):
        plt.imsave(likely_path+f'\Likely {i+1}.png'.format(int()), likely_img)
    else:
        plt.imsave(likely_path+f'\Likely {i}.png', likely_img)

# Likely to Undergo RI files are processed
print('''"Likely to Undergo RI" files have been processed''')

# resetting file naming variable
i=0

# Not Likely to Undergo RI
for f in notL_csv_files:
    # read the csv file
    notL_df = pd.read_csv(f, header=None)

    # Converting pandas dataframes to numpy arrays
    notL_arr = notL_df.to_numpy()

    # Converting NaN values to 0
    notL_farray = np.nan_to_num(notL_arr, nan=0.0)

    # Converting float array to int array
    notL_farray = np.around(notL_farray)
    notL_iarray = notL_farray.astype(int)

    # converting array to image
    w, h = 300, 300
    notL_img = [np.zeros((h, w)).astype(int), np.zeros((h, w)).astype(int), np.zeros((h, w)).astype(int)]
    notL_img[0:300] = notL_iarray
    notL_img = Image.fromarray(notL_iarray)

    # saving images
    notL_path = "G:\Python Projects\Science Research Project\MIMIC-TC Dataset v3\Training Set\Likely Not to Undergo RI"
    notL_files = os.listdir(notL_path)
    i = i+1

    if os.path.exists(notL_path+f"\LikelyNot {i}.png"):
        plt.imsave(notL_path+f'\LikelyNot {i+1}.png'.format(int()), notL_img)
    else:
        plt.imsave(notL_path+f'\LikelyNot {i}.png', notL_img)

# Not Likely to Undergo RI files are processed
print('''"Not Likely to Undergo RI" files have been processed''')

# loading train images into keras
RI_dataset = tf.keras.preprocessing.image_dataset_from_directory('G:\Python Projects\Science Research Project\MIMIC-TC Dataset v3', image_size=(300,300), batch_size=32)

print(type(RI_dataset))
