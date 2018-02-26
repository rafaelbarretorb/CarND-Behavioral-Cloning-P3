#  cd C:\Users\rafael\Desktop\Udacity\SDCN\P3\CarND-Behavioral-Cloning-P3-master

#  python model_no_generator.py model.h5

#  python drive.py model.h5


import numpy as np
import pandas as pd
import csv
import cv2
import os
import sklearn
import random
from random import randrange
from random import shuffle

from keras.backend import tf



import h5py

import sys

lines = []
with open('Records/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images_center = [] # input labels
measurements = []  # output labels
for line in lines:
    img_file = line[0]
    img_file = img_file.split('\\')[-1]
    current_path = 'Records/IMG/' + img_file
    image = cv2.imread(current_path)
    images_center.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

### DATA AUGMENTATION

# mirror images
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images_center, measurements):
	augmented_images.append(image)
	augmented_measurements.append(measurement)
	augmented_images.append(np.fliplr(image))
    #augmented_images.append(cv.flip(image,1))
	augmented_measurements.append(measurement*(-1.0))

X_train_full = np.array(augmented_images)
y_train_full = np.array(augmented_measurements)

### DATA REDUCTION
# Exclude part of the data that has angles near to zero avoiding a model with bias associated with going straight

near = 0.06
count = 0
for i in y_train_full:
    if np.absolute(i) <= near:
        count = count + 1


inds_near_to_zero = [] # List of indices with measument near to zero
for i,ii in enumerate(y_train_full):
    if np.absolute(ii) <= near:
        inds_near_to_zero.append(i)

from random import randrange

# List of indices to delete
percent_to_delete = 0.95
n = int(count*percent_to_delete) # number of itens to delete

n_numbers_to_remove = len(inds_near_to_zero) - n

for i in range(n_numbers_to_remove):
    random_index = randrange(0,len(inds_near_to_zero))
    inds_near_to_zero.pop(random_index)

X_train = np.delete(X_train_full, inds_near_to_zero, axis=0)
y_train = np.delete(y_train_full, inds_near_to_zero)




from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras import optimizers
from keras.models import load_model
from keras.layers.core import Dropout


if len(sys.argv) == 2:
    model = load_model('model.h5')
    print ("")
    print ("Weights loaded!")
    print ("")
else:
    model = Sequential()
    print ("")
    print ("Sequential!!!")
    print ("")

    model.add(Lambda(lambda x: x/127.5 - 1, input_shape=(160, 320,3)))
    model.add(Cropping2D(cropping=((70,20),(0,0))))


    #model.add(Lambda(lambda x:  tf.image.resize_images(x, (80, 160)), input_shape=(3, 160, 320), output_shape=(3,80,160))) # does not work


    # NVIDIA
    model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="relu"))
    model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="relu"))
    model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Flatten())
    model.add(Dense(1164))
    model.add(Dense(100))
    model.add(Dropout(0.6))
    model.add(Dense(50))
    model.add(Dense(10, activation="relu"))
    model.add(Dense(1))

adam = optimizers.Adam(lr=0.00001)
model.compile(loss='mse',optimizer=adam)
#model.compile(loss='mse',optimizer='adam')


model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=3)


model.save('model.h5')
print ("")
print ("Model saved!!")
print ("")
