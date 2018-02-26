#  cd C:\Users\rafael\Desktop\Udacity\SDCN\P3\CarND-Behavioral-Cloning-P3-master

#  python model.py model.h5

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
from random import randrange

import h5py

import sys


samples = []
with open('Records/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)


from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

batch_size=32
def generator(samples, batch_size=batch_size):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = 'Records/IMG/' + batch_sample[0].split('\\')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            ### DATA AUGMENTATION
            # mirror images
            augmented_images, augmented_angles = [], []
            for image, angle in zip(images, angles):
                augmented_images.append(image)
                augmented_angles.append(angle)
                augmented_images.append(np.fliplr(image))
                augmented_angles.append(angle*(-1.0))

            X_train_full = np.array(augmented_images)
            y_train_full = np.array(augmented_angles)

            ### DATA REDUCTION

            # Count the number of measurements near to zero
            near = 0.06
            count = 0
            for i in y_train_full:
                if np.absolute(i) <= near:
                    count = count + 1

            inds_near_to_zero = [] # List of indices with measument near to zero
            for i,ii in enumerate(y_train_full):
                if np.absolute(ii) <= near:
                    inds_near_to_zero.append(i)

            # List of indices to delete
            percent_to_delete = 0.85
            n = int(count*percent_to_delete) # number of itens to delete

            n_numbers_to_remove = len(inds_near_to_zero) - n

            for i in range(n_numbers_to_remove):
                random_index = randrange(0,len(inds_near_to_zero))
                inds_near_to_zero.pop(random_index)


            X_train = np.delete(X_train_full, inds_near_to_zero, axis=0)
            y_train = np.delete(y_train_full, inds_near_to_zero)


            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


### CNN ARCHITECTURE 


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

adam = optimizers.Adam(lr=0.0001)
model.compile(loss='mse',optimizer=adam)


model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), epochs=2)


model.save('model.h5')

model.save('model.h5')
print ("")
print ("Model saved!!")
print ("")
