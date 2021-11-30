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
notL_csv_files = glob.glob(os.path.join(r"D:\Science Research 2\MIMIC-TC Dataset v3\Training Dataset\No RI","*.csv"))

# variable for file renaming
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
    notL_path = r'G:\Python Projects\Science Research Project\MIMIC-TC Dataset v3\Training Set\No RI'
    notL_files = os.listdir(notL_path)
    i = i+1

    if os.path.exists(notL_path+fr"\nRI {i}.png"):
        plt.imsave(notL_path+fr'\nRI {i+1}.png'.format(int()), notL_img)
    else:
        plt.imsave(notL_path+fr'\nRI {i}.png', notL_img)