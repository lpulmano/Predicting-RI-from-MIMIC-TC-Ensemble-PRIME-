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
training_csv_files = glob.glob(os.path.join(r"D:\Science Research 2\MIMIC-TC Dataset\Training Dataset\Lorenzo\2019 09 29 06h.csv"))

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

    # print the location and filename
    print('Location:', f)
    print('File Name:', f.split("\\")[-1])

    # print out the content
    print('Content:')
    display(training_iarray)
    print()

    # converting array to image
    w, h = 300, 300
    training_img = [np.zeros((h, w)).astype(int), np.zeros((h, w)).astype(int), np.zeros((h, w)).astype(int)]
    training_img[0:300] = training_iarray
    training_img = Image.fromarray(training_iarray)
    training_img.show()

    testing_path = 'G:\Python Projects\Science Research Project\Lorenzo Images'
    plt.imsave('G:\Python Projects\Science Research Project\Lorenzo Images\lorenzo test.png', training_img)

# plt.imsave('G:\Python Projects\Science Research Project\Lorenzo Images\*.png', training_img)




# moving and renaming images
# testing_path = 'G:\Python Projects\Science Research Project\Lorenzo Images'
# testing_files = os.listdir(testing_path)
# i = 1

# for index, file in enumerate(testing_files):
    # os.rename(os.path.join(testing_path, file), os.path.join(testing_path, 'Lorenzo '+str(i)+'.png'))