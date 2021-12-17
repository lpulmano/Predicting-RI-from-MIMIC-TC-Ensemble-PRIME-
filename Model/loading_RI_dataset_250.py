import numpy as np
import os
import cv2

# Insert the file path of the training dataset
print("Type the exact file path of the MIMIC-TC v4 Dataset.")
print("Example: G:\Python Projects\Science Research Project")
dataset_file_path = input()
dataset_file_path = fr"{dataset_file_path}/MIMIC-TC Dataset v4/Training Set"

# Categories
CATEGORIES = ["Current RI","Possible RI","No RI"]

# creating training data
training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(dataset_file_path, category)  # path to RI classifications
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (250,250))
            training_data.append([new_array, class_num])

create_training_data()
print(len(training_data))

# shuffling data
import random
random.shuffle(training_data)

for sample in training_data[:10]:
    print(sample[1])

# features and labels
X_RI = []
y_RI = []

for features, label in training_data:
    X_RI.append(features)
    y_RI.append(label)

# reshaping
X_RI = np.array(X_RI).reshape(-1, 250, 250, 1)

import pickle

pickle_out = open("X_RI_250.pickle","wb")
pickle.dump(X_RI, pickle_out)
pickle_out.close()

pickle_out = open("y_RI_250.pickle","wb")
pickle.dump(y_RI, pickle_out)
pickle_out.close()

pickle_in = open("X_RI_250.pickle","rb")
X_RI = pickle.load(pickle_in)

# for category in CATEGORIES:
#     path = os.path.join(dataset_file_path, category)  # path to RI classifications
#     for img in os.listdir(path):
#         img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
#         plt.imshow(img_array, cmap="gray")
#         plt.show()
#         break
#     break


# IMG_SIZE = 300
#
# new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
# plt.imshow(new_array, cmap='gray')