import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
import datetime

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# RI Dataset (includes Current RI, Possible RI, No RI)
from loading_RI_dataset_4 import dataset_file_path
from loading_RI_dataset_4 import X_RI, y_RI
X_RI = pickle.load(open("X_RI.pickle","rb"))
y_RI = pickle.load(open("y_RI.pickle","rb"))

X_RI = X_RI/255.0



# creating the model
model = Sequential()

model.add(layers.Conv2D(64, (5,5), activation='relu', input_shape=X_RI.shape[1:]))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64, (5,5), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.BatchNormalization())

model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))

#model.add(layers.Conv2D(64, (3,3), activation='relu'))
#model.add(layers.Conv2D(64, (3,3), activation='relu'))
#model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.Conv2D(64, (2,2), activation='relu'))
model.add(Flatten())

model.add(layers.Dense(1000, activation='relu'))
model.add(layers.Dense(200, activation='relu'))
model.add(layers.Dense(1, activation='softmax'))

# compiling model
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])

X_RI = np.array(X_RI).reshape(-1, 300, 300, 1)
y_RI = np.array(y_RI)

# directory for Tensorboard log
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# fitting the model
model.fit(X_RI, y_RI, epochs=30, batch_size=32, validation_split=0.3, callbacks=[tensorboard_callback])

# saving the model
model.save('RI_CNN_v10.model')

# Tensorboard
# go to Terminal
# tensorboard --logdir=Model/logs/
