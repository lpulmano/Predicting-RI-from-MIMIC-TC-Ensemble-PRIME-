# import necessary libraries
from IPython.display import display
import pandas as pd
import os
import glob


# use glob to get all the csv files in the folder
path = os.getcwd()
csv_files = glob.glob(os.path.join("D:\Science Research 2\MIMIC-TC Dataset\Training Dataset\Lorenzo", "*.csv"))

# loop over the list of csv files
for f in csv_files:

    # read the csv file
    df = pd.read_csv(f, header= None)

    # print the location and filename
    print('Location:', f)
    print('File Name:', f.split("\\")[-1])

    # print out the content
    print('Content:')
    display(df)
    print()