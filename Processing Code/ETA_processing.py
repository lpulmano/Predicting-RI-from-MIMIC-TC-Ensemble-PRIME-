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

# variable for file renaming
i=0

# ETA csv processing
test_csv_files = glob.glob(os.path.join(r"D:\Science Research 2\MIMIC-TC Dataset v3\Testing Dataset\ETA\Current RI","*.csv"))
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
    test_path = r'G:\Python Projects\Science Research Project\MIMIC-TC Dataset v3\Testing Set\ETA\Current RI'
    test_files = os.listdir(test_path)
    i = i+1

    if os.path.exists(test_path+fr"\ETA cRI {i}.png"):
        plt.imsave(test_path+fr'\ETA cRI {i+1}.png'.format(int()), test_img)
    else:
        plt.imsave(test_path+fr'\ETA cRI {i}.png', test_img)

# resetting file naming variable
i=0

test_csv_files = glob.glob(os.path.join(r"D:\Science Research 2\MIMIC-TC Dataset v3\Testing Dataset\ETA\Possible RI","*.csv"))
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
    test_path = r'G:\Python Projects\Science Research Project\MIMIC-TC Dataset v3\Testing Set\ETA\Possible RI'
    test_files = os.listdir(test_path)
    i = i+1

    if os.path.exists(test_path+fr"\ETA pRI {i}.png"):
        plt.imsave(test_path+fr'\ETA pRI {i+1}.png'.format(int()), test_img)
    else:
        plt.imsave(test_path+fr'\ETA pRI {i}.png', test_img)

# resetting file naming variable
i=0

test_csv_files = glob.glob(os.path.join(r"D:\Science Research 2\MIMIC-TC Dataset v3\Testing Dataset\ETA\No RI","*.csv"))
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
    test_path = r'G:\Python Projects\Science Research Project\MIMIC-TC Dataset v3\Testing Set\ETA\No RI'
    test_files = os.listdir(test_path)
    i = i+1

    if os.path.exists(test_path+f"\ETA nRI {i}.png"):
        plt.imsave(test_path+fr'\ETA nRI {i+1}.png'.format(int()), test_img)
    else:
        plt.imsave(test_path+fr'\ETA nRI {i}.png', test_img)

# resetting file naming variable
i=0