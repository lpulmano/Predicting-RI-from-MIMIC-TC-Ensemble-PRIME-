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

# Insert the file path of the dataset
print("Type the exact file path of the MIMIC-TC v3 Dataset.")
print("Example: G:\Python Projects\Science Research Project")
dataset_file_path = input()

# loading train images into keras
RI_dataset = tf.keras.preprocessing.image_dataset_from_directory(fr'{dataset_file_path}\MIMIC-TC Dataset v3\Training Set', image_size=(300,300), batch_size=32)
print("Loaded RI Dataset!")
print(type(RI_dataset))

# loading all storm images into keras (44 individual datasets)
ALPHA_dataset = tf.keras.preprocessing.image_dataset_from_directory(fr'{dataset_file_path}\MIMIC-TC Dataset v3\Testing Set\ALPHA', image_size=(300,300), batch_size=32)
print("Loaded ALPHA Dataset!")
Arthur_dataset = tf.keras.preprocessing.image_dataset_from_directory(fr'{dataset_file_path}\MIMIC-TC Dataset v3\Testing Set\Arthur', image_size=(300,300), batch_size=32)
print("Loaded Arthur Dataset!")
Bertha_dataset = tf.keras.preprocessing.image_dataset_from_directory(fr'{dataset_file_path}\MIMIC-TC Dataset v3\Testing Set\Bertha', image_size=(300,300), batch_size=32)
print("Loaded Bertha Dataset!")
BETA_dataset = tf.keras.preprocessing.image_dataset_from_directory(fr'{dataset_file_path}\MIMIC-TC Dataset v3\Testing Set\BETA', image_size=(300,300), batch_size=32)
print("Loaded BETA Dataset!")
Cristobal_dataset = tf.keras.preprocessing.image_dataset_from_directory(fr'{dataset_file_path}\MIMIC-TC Dataset v3\Testing Set\Cristobal', image_size=(300,300), batch_size=32)
print("Loaded Cristobal Dataset!")
DETLA_dataset = tf.keras.preprocessing.image_dataset_from_directory(fr'{dataset_file_path}\MIMIC-TC Dataset v3\Testing Set\DELTA', image_size=(300,300), batch_size=32)
print("Loaded DELTA Dataset!")
Dolly_dataset = tf.keras.preprocessing.image_dataset_from_directory(fr'{dataset_file_path}\MIMIC-TC Dataset v3\Testing Set\Dolly', image_size=(300,300), batch_size=32)
print("Loaded Dolly Dataset!")
Dorian_dataset = tf.keras.preprocessing.image_dataset_from_directory(fr'{dataset_file_path}\MIMIC-TC Dataset v3\Testing Set\Dorian', image_size=(300,300), batch_size=32)
print("Loaded Dorian Dataset!")
Edouard_dataset = tf.keras.preprocessing.image_dataset_from_directory(fr'{dataset_file_path}\MIMIC-TC Dataset v3\Testing Set\Edouard', image_size=(300,300), batch_size=32)
print("Loaded Edouard Dataset!")
EPSILON_dataset = tf.keras.preprocessing.image_dataset_from_directory(fr'{dataset_file_path}\MIMIC-TC Dataset v3\Testing Set\EPSILON', image_size=(300,300), batch_size=32)
print("Loaded EPSILON Dataset!")
ETA_dataset = tf.keras.preprocessing.image_dataset_from_directory(fr'{dataset_file_path}\MIMIC-TC Dataset v3\Testing Set\ETA', image_size=(300,300), batch_size=32)
print("Loaded ETA Dataset!")
Fay_dataset = tf.keras.preprocessing.image_dataset_from_directory(fr'{dataset_file_path}\MIMIC-TC Dataset v3\Testing Set\Fay', image_size=(300,300), batch_size=32)
print("Loaded Fay Dataset!")
FIFTEEN2019_dataset = tf.keras.preprocessing.image_dataset_from_directory(fr'{dataset_file_path}\MIMIC-TC Dataset v3\Testing Set\FIFTEEN - 2019', image_size=(300,300), batch_size=32)
print("Loaded FIFTEEN2019 Dataset!")
GAMMA_dataset = tf.keras.preprocessing.image_dataset_from_directory(fr'{dataset_file_path}\MIMIC-TC Dataset v3\Testing Set\GAMMA', image_size=(300,300), batch_size=32)
print("Loaded GAMMA Dataset!")
Gonzalo_dataset = tf.keras.preprocessing.image_dataset_from_directory(fr'{dataset_file_path}\MIMIC-TC Dataset v3\Testing Set\Gonzalo', image_size=(300,300), batch_size=32)
print("Loaded Gonzalo Dataset!")
Hanna_dataset = tf.keras.preprocessing.image_dataset_from_directory(fr'{dataset_file_path}\MIMIC-TC Dataset v3\Testing Set\Hanna', image_size=(300,300), batch_size=32)
print("Loaded Hanna Dataset!")
Humberto_dataset = tf.keras.preprocessing.image_dataset_from_directory(fr'{dataset_file_path}\MIMIC-TC Dataset v3\Testing Set\Humberto', image_size=(300,300), batch_size=32)
print("Loaded Humberto Dataset!")
Imelda_dataset = tf.keras.preprocessing.image_dataset_from_directory(fr'{dataset_file_path}\MIMIC-TC Dataset v3\Testing Set\Imelda', image_size=(300,300), batch_size=32)
print("Loaded Imelda Dataset!")
IOTA_dataset = tf.keras.preprocessing.image_dataset_from_directory(fr'{dataset_file_path}\MIMIC-TC Dataset v3\Testing Set\IOTA', image_size=(300,300), batch_size=32)
print("Loaded IOTA Dataset!")
Isaias_dataset = tf.keras.preprocessing.image_dataset_from_directory(fr'{dataset_file_path}\MIMIC-TC Dataset v3\Testing Set\Isaias', image_size=(300,300), batch_size=32)
print("Loaded Isaias Dataset!")
Jerry_dataset = tf.keras.preprocessing.image_dataset_from_directory(fr'{dataset_file_path}\MIMIC-TC Dataset v3\Testing Set\Jerry', image_size=(300,300), batch_size=32)
print("Loaded Jerry Dataset!")
Josephine_dataset = tf.keras.preprocessing.image_dataset_from_directory(fr'{dataset_file_path}\MIMIC-TC Dataset v3\Testing Set\Josephine', image_size=(300,300), batch_size=32)
print("Loaded Josephine Dataset!")
Karen_dataset = tf.keras.preprocessing.image_dataset_from_directory(fr'{dataset_file_path}\MIMIC-TC Dataset v3\Testing Set\Karen', image_size=(300,300), batch_size=32)
print("Loaded Karen Dataset!")
Kyle_dataset = tf.keras.preprocessing.image_dataset_from_directory(fr'{dataset_file_path}\MIMIC-TC Dataset v3\Testing Set\Kyle', image_size=(300,300), batch_size=32)
print("Loaded Kyle Dataset!")
Laura_dataset = tf.keras.preprocessing.image_dataset_from_directory(fr'{dataset_file_path}\MIMIC-TC Dataset v3\Testing Set\Laura', image_size=(300,300), batch_size=32)
print("Loaded Laura Dataset!")
Lorenzo_dataset = tf.keras.preprocessing.image_dataset_from_directory(fr'{dataset_file_path}\MIMIC-TC Dataset v3\Testing Set\Lorenzo', image_size=(300,300), batch_size=32)
print("Loaded Lorenzo Dataset!")
Marco_dataset = tf.keras.preprocessing.image_dataset_from_directory(fr'{dataset_file_path}\MIMIC-TC Dataset v3\Testing Set\Marco', image_size=(300,300), batch_size=32)
print("Loaded Marco Dataset!")
Melissa_dataset = tf.keras.preprocessing.image_dataset_from_directory(fr'{dataset_file_path}\MIMIC-TC Dataset v3\Testing Set\Melissa', image_size=(300,300), batch_size=32)
print("Loaded Melissa Dataset!")
Nana_dataset = tf.keras.preprocessing.image_dataset_from_directory(fr'{dataset_file_path}\MIMIC-TC Dataset v3\Testing Set\Nana', image_size=(300,300), batch_size=32)
print("Loaded Nana Dataset!")
Nestor_dataset = tf.keras.preprocessing.image_dataset_from_directory(fr'{dataset_file_path}\MIMIC-TC Dataset v3\Testing Set\Nestor', image_size=(300,300), batch_size=32)
print("Loaded Nestor Dataset!")
Olga_dataset = tf.keras.preprocessing.image_dataset_from_directory(fr'{dataset_file_path}\MIMIC-TC Dataset v3\Testing Set\Olga', image_size=(300,300), batch_size=32)
print("Loaded Olga Dataset!")
Omar_dataset = tf.keras.preprocessing.image_dataset_from_directory(fr'{dataset_file_path}\MIMIC-TC Dataset v3\Testing Set\Omar', image_size=(300,300), batch_size=32)
print("Loaded Omar Dataset!")
Pablo_dataset = tf.keras.preprocessing.image_dataset_from_directory(fr'{dataset_file_path}\MIMIC-TC Dataset v3\Testing Set\Pablo', image_size=(300,300), batch_size=32)
print("Loaded Pablo Dataset!")
Paulette_dataset = tf.keras.preprocessing.image_dataset_from_directory(fr'{dataset_file_path}\MIMIC-TC Dataset v3\Testing Set\Paulette', image_size=(300,300), batch_size=32)
print("Loaded Paulette Dataset!")
Rebekah_dataset = tf.keras.preprocessing.image_dataset_from_directory(fr'{dataset_file_path}\MIMIC-TC Dataset v3\Testing Set\Rebekah', image_size=(300,300), batch_size=32)
print("Loaded Rebekah Dataset!")
Rene_dataset = tf.keras.preprocessing.image_dataset_from_directory(fr'{dataset_file_path}\MIMIC-TC Dataset v3\Testing Set\Rene', image_size=(300,300), batch_size=32)
print("Loaded Rene Dataset!")
Sally_dataset = tf.keras.preprocessing.image_dataset_from_directory(fr'{dataset_file_path}\MIMIC-TC Dataset v3\Testing Set\Sally', image_size=(300,300), batch_size=32)
print("Loaded Sally Dataset!")
Sebastien_dataset = tf.keras.preprocessing.image_dataset_from_directory(fr'{dataset_file_path}\MIMIC-TC Dataset v3\Testing Set\Sebastien', image_size=(300,300), batch_size=32)
print("Loaded Sebastien Dataset!")
Teddy_dataset = tf.keras.preprocessing.image_dataset_from_directory(fr'{dataset_file_path}\MIMIC-TC Dataset v3\Testing Set\Teddy', image_size=(300,300), batch_size=32)
print("Loaded Teddy Dataset!")
TEN2020_dataset = tf.keras.preprocessing.image_dataset_from_directory(fr'{dataset_file_path}\MIMIC-TC Dataset v3\Testing Set\TEN - 2020', image_size=(300,300), batch_size=32)
print("Loaded TEN2020 Dataset!")
THETA_dataset = tf.keras.preprocessing.image_dataset_from_directory(fr'{dataset_file_path}\MIMIC-TC Dataset v3\Testing Set\THETA', image_size=(300,300), batch_size=32)
print("Loaded THETA Dataset!")
Vicky_dataset = tf.keras.preprocessing.image_dataset_from_directory(fr'{dataset_file_path}\MIMIC-TC Dataset v3\Testing Set\Vicky', image_size=(300,300), batch_size=32)
print("Loaded Vicky Dataset!")
Wilfred_dataset = tf.keras.preprocessing.image_dataset_from_directory(fr'{dataset_file_path}\MIMIC-TC Dataset v3\Testing Set\Wilfred', image_size=(300,300), batch_size=32)
print("Loaded Wilfred Dataset!")
ZETA_dataset = tf.keras.preprocessing.image_dataset_from_directory(fr'{dataset_file_path}\MIMIC-TC Dataset v3\Testing Set\ZETA', image_size=(300,300), batch_size=32)
print("Loaded ZETA Dataset!")

print("Loaded all storm datasets!")