# MIMIC-TC-Dataset-v3-NEW

"Processing Code" folder is simply a collection of scripts I created to convert the respective csv files for each storms into their images.
These scripts are not needed for the CNN to run.  Only the produced images are required, which will be linked as a Google Drive download (https://drive.google.com/drive/folders/1oGStvodx2EUwYsD23Xc81c1yR25-qdtZ?usp=sharing)

"Practice files" folder contains relics in both "Processing Code" and "Model" folders.  It shows my first attempts at reading .csv data, 
showing images using matplotlib, loading datasets, and more.  It is not required for the model to run.

"Model" folder currently contains my attempts at loading the RI dataset and running the DeepMicroNet model architecture.
The different "model *.py" files are separate attempts of trying out new code; they are similar in some aspects but for the most part, trying to load the datasets is different for each.

Nomenclature:

cRI = Current RI

pRI = Possible RI

nRI = No RI

The images in the above categories have not been reviewed if they best fit RI criteria/satellite images as of 12/13/21 (December 13 2021).

HAS BEEN RESOLVED
{I do not know how to load "Training Set" and its contents as the training data for the model.
I do not know how to load the individual storm directories in "Testing Set" as testing data for the model.

I am having great difficulty in loading my custom datasets into the model.}
HAS BEEN RESOLVED

File paths are in the Windows OS format.

Tensorboard Command:

tensorboard --logdir Model/logs/

Pip installations:

pip install scikit-image

pip install pandas

pip install scikit-learn

pip install opencv-python

pip install split-folders

pip install tensorflow

pip install seaborn

pip install ipython

pip install matplotlib

pip install jupyterlab

pip install jupyter
