import pandas as pd
import numpy as np
import os
import glob
from IPython.display import display


# use glob to get all the csv files in the folder
path = os.getcwd()
training_csv_files = glob.glob(os.path.join("D:\Science Research 2\MIMIC-TC Dataset\Training Dataset\Lorenzo","*.csv"))
testing_csv_files = glob.glob(os.path.join("D:\Science Research 2\MIMIC-TC Dataset\Testing Dataset\Arthur","*.csv"))

for f in training_csv_files:
    # read the csv file
    training_df = pd.read_csv(f, header= None)

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

    # print the location and filename
    print('Location:', f)
    print('File Name:', f.split("\\")[-1])

    # print out the content
    print('Content:')
    display(testing_iarray)
    print()
