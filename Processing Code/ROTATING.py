import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# Insert the file path of the testing dataset
print("Type the RI classification.")
classification = input()

# png dataset filepath
png_files = glob.glob(os.path.join(fr"G:\Python Projects\Science Research Project\MIMIC-TC Dataset v4\Training Set\{classification}","*.png"))

# variable for file renaming
i=0

# blackslash replacement
backslash = "\\"

for f in png_files:
    # rotating image
    im = Image.open(f)
    im = im.transpose(Image.ROTATE_90)
    # saving images
    test_path = fr"G:\Python Projects\Science Research Project\MIMIC-TC Dataset v4\Training Set\{classification}"
    test_files = os.listdir(test_path)

    if os.path.exists(fr'''{test_path}/{Path(f.split(fr'{backslash}')[-1]).stem}.png'''):
        plt.imsave(fr'''{test_path}/{Path(f.split(fr'{backslash}')[-1]).stem} ROT.png'''.format(int()), im)
    else:
        plt.imsave(fr'''{test_path}/{Path(f.split(fr'{backslash}')[-1]).stem}.png'''.format(int()), im)
