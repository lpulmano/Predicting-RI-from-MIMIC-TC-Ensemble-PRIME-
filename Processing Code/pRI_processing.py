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
# likely csv files
likely_csv_files = glob.glob(os.path.join(r"D:\Science Research 2\MIMIC-TC Dataset v3\Training Dataset\Possible RI","*.csv"))

# variable for file renaming
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