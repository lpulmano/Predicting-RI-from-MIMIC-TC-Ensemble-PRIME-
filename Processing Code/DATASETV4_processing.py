import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# Insert the file path of the training dataset
print("Type the RI category.")
category = input()

# csv dataset filepath
csv_files = glob.glob(os.path.join(fr"D:\Science Research 2\MIMIC-TC Dataset v4\Training Dataset/{category}","*.csv"))

# Image dataset filepath
category_folder = os.mkdir(fr"G:\Python Projects\Science Research Project\MIMIC-TC Dataset v4\Training Set/{category}")

# variable for file renaming
i=0

# blackslash replacement
backslash = "\\"

for f in csv_files:

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
    test_path = fr"G:\Python Projects\Science Research Project\MIMIC-TC Dataset v4\Training Set/{category}"
    test_files = os.listdir(test_path)

    plt.imsave(fr'''{test_path}/{Path(f.split(fr'{backslash}')[-1]).stem}.png'''.format(int()), test_img)

