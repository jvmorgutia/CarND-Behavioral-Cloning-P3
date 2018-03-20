import csv
import cv2
import numpy as np
import tensorflow as tf

#import matplotlib as mpl
#mpl.use('TkAgg')
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

def append_image(filename, adjustment):
    #if float(line[3]) > 0.01:
    current_path = '/data/IMG/' + filename_center
    image = cv2.imread(current_path)
    flipped = cv2.flip(image,1)
    images.append(image)
    images.append(flipped)
    measurement = float(line[3]) + float(adjustment)
    measurements.append(measurement)
    measurements.append(measurement*-1.0)

    
def preprocess(images):
    batch = np.zeros((images.shape[0],160,320,3))
    for img in range(images.shape[0]):
        batch[img] = img
    return batch

def train():
    model = Sequential()
    model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((60, 20), (0,0))))

    model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(64,3,3,activation="relu"))
    model.add(Convolution2D(64,3,3,activation="relu"))

    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(150))
    model.add(Dense(10))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer ='adam')
    model.fit(x=x_processed, y=y_train, validation_split=0.2, shuffle=True, epochs=3)

    model.save('model.h5')
    print('Finished saving.')


lines = []
images = []
measurements = [] 

with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
    
for line in lines:
    source_center = line[0]
    #source_left = line[1]
    #source_right=line[2]
    filename_center = source_center.split('/')[-1]
    #filename_left = source_left.split('/')[-1]
    #filename_right = source_right.split('/')[-1]    
    append_image(filename_center,0)
    #append_image(filename_left, 0.25)
    #append_image(filename_right,-0.25)

print('Gatheing Data...')   

X_train = np.array(images)
y_train = np.array(measurements)
print(X_train.shape)
#print(X_train[0])
#plt.imshow(X_train[0])

print('Processing Data...')

x_processed = preprocess(X_train)
print('Starting Training...')
print()

train()


