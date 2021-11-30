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