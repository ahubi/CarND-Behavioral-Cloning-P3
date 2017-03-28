import os
import csv
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D
from sklearn.model_selection import train_test_split
import random
import sys
import matplotlib.pyplot as plt
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
                name = './' + datadir + '/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                if flip_images==1:
                    images.append(cv2.flip(center_image, 1))
                    angles.append(center_angle*-1.0)

            # provide shuffled train data
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

#plots history object of the model and stores plot to file
def plot_history_model(history_object, stor2file=None):
    ### print the keys contained in the history object
    print(history_object.history.keys())

    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    if(stor2file!=None):
        plt.savefig(stor2file)
    plt.show()

#parse directory with images, a directory can be provided as argument
datadir = 'data'
if len(sys.argv) > 1:
    datadir = str(sys.argv[1])
    print ('Using images from:', datadir)

samples = []
with open('./' + datadir + '/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    i = 0
    for line in reader:
        if i > 0: #skip first line which is naming
            samples.append(line)
        i += 1

#split train and validation samples 80/20%
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32, flip_images=1)
validation_generator = generator(validation_samples, batch_size=32, flip_images=0)
#image format
ch, row, col = 3, 160, 320
# Use Nvidia model
model = Sequential()
#add normalization to network
model.add(Lambda(lambda x: (x / 255.0) - 0.5,
                input_shape=(row, col, ch),
                output_shape=(row, col, ch)))
#add cropping to remove unrelevant information in the pictures
#remove 70 pixel from above, and 25 from below
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(row, col, ch)))
# Nvidia architecture
# https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
model.add(Convolution2D(24,5,5, subsample=(2,2), activation = 'relu'))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation = 'relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation = 'relu'))
model.add(Convolution2D(64,3,3, activation = 'relu'))
model.add(Convolution2D(64,3,3, activation = 'relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
hobj = model.fit_generator(train_generator, samples_per_epoch=len(train_samples), \
                    validation_data=validation_generator, \
                    nb_val_samples=len(validation_samples), nb_epoch=15)
#save trained model to a file for later use in simulator
model.save('model.h5')
#plot history object,
#this line is to be deactivated on aws, since it doesn't work there
plot_history_model(hobj, 'mse_model_loss_9_epochs.png')
