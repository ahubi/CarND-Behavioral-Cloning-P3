import os
import csv
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
import random

samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    i = 0
    for line in reader:
        if i > 0: #skip first line which is naming
            samples.append(line)
        i += 1


from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn

def generator(samples, batch_size=32, flip_images=1):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                if flip_images==1:
                    images.append(cv2.flip(center_image, 1))
                    angles.append(center_angle*-1.0)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32, flip_images=1)
validation_generator = generator(validation_samples, batch_size=32, flip_images=0)

ch, row, col = 3, 160, 320  # Trimmed image format
model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1.,
                input_shape=(row, col, ch),
                output_shape=(row, col, ch)))
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(row, col, ch)))
model.add(Flatten(input_shape=(row, col, ch)))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), \
                    validation_data=validation_generator, \
                    nb_val_samples=len(validation_samples), nb_epoch=3)
model.save('model.h5')
