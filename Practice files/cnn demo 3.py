import pandas as pd
import numpy as np
import os
import glob
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
from PIL import Image


# use glob to get all the csv files in the folder
path = os.getcwd()
training_csv_files = glob.glob(os.path.join("D:\Science Research 2\MIMIC-TC Dataset\Training Dataset\Lorenzo","*.csv"))
testing_csv_files = glob.glob(os.path.join("D:\Science Research 2\MIMIC-TC Dataset\Testing Dataset\Arthur","*.csv"))

# variable for file renaming
i=0

# Reading, converting, and saving the training files
for f in training_csv_files:
    # read the csv file
    training_df = pd.read_csv(f, header=None)

    # Converting pandas dataframes to numpy arrays
    training_arr = training_df.to_numpy()

    # Converting NaN values to 0
    training_farray = np.nan_to_num(training_arr, nan=0.0)

    # Converting float array to int array
    training_farray = np.around(training_farray)
    training_iarray = training_farray.astype(int)

    # converting array to image
    w, h = 300, 300
    training_img = [np.zeros((h, w)).astype(int), np.zeros((h, w)).astype(int), np.zeros((h, w)).astype(int)]
    training_img[0:300] = training_iarray
    training_img = Image.fromarray(training_iarray)

    # saving images
    training_path = 'G:\Python Projects\Science Research Project\Training Images'
    training_files = os.listdir(training_path)
    i = i+1

    if os.path.exists(training_path+f'\Lorenzo {i}.png'):
        plt.imsave(training_path+f'\Lorenzo {i+1}.png'.format(int()), training_img)
    else:
        plt.imsave(training_path+f'\Lorenzo {i}.png', training_img)

# resetting file naming variable
i=0

# Reading, converting, and saving the testing files
for f in testing_csv_files:
    # read the csv file
    testing_df = pd.read_csv(f, header=None)

    # Converting pandas dataframes to numpy arrays
    testing_arr = testing_df.to_numpy()

    # Converting NaN values to 0
    testing_farray = np.nan_to_num(testing_arr, nan=0.0)

    # Converting float array to int array
    testing_farray = np.around(testing_farray)
    testing_iarray = testing_farray.astype(int)

    # converting array to image
    w, h = 300, 300
    testing_img = [np.zeros((h, w)).astype(int), np.zeros((h, w)).astype(int), np.zeros((h, w)).astype(int)]
    testing_img[0:300] = testing_iarray
    testing_img = Image.fromarray(testing_iarray)

    # saving images
    testing_path = 'G:\Python Projects\Science Research Project\Testing Images'
    testing_files = os.listdir(testing_path)
    i = i+1

    if os.path.exists(testing_path+f'\Arthur {i}.png'):
        plt.imsave(testing_path+f'\Arthur {i+1}.png'.format(int()), testing_img)
    else:
        plt.imsave(testing_path+f'\Arthur {i}.png', testing_img)
