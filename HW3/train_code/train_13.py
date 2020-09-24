import os
import sys
import numpy as np
import pandas as pd
import csv
import random as rand
import math

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import BatchNormalization, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam, RMSprop, Adagrad, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
#from keras.regularizers import l1,l2,l1l2

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# File
input_1     = sys.argv[1]           # train.csv

# Parameter
row = 28709         # 28709 Data
input_size = 48     # 48 x 48 pixels

# Open train.csv file
data_x = pd.read_csv(input_1, encoding = 'big5')
np_data_x = np.array(data_x)
np_data_x = np_data_x.tolist()  

# reshape the train data and label_data
train_data = list()
label_data = list()
for i in range(row):
    a = np_data_x[i][1].split()
    b = np.array([float(j) for j in a])
    c = b.reshape((48,48))
    c = c.tolist()

    train_data.append(c)
    label_data.append(np_data_x[i][0])

#Split Validation
split = np.random.permutation(row)
x_data = list()
y_data = list()
v_x_data = list()
v_y_data = list()

for i in range(row):
    if (i<25839):
        x_data.append(train_data[split[i]])
        y_data.append(label_data[split[i]])
    else:
        v_x_data.append(train_data[split[i]])
        v_y_data.append(label_data[split[i]])

x_data = np.array(x_data)
y_data = np.array(y_data)
v_x_data = np.array(v_x_data)
v_y_data = np.array(v_y_data)
x_data = np.expand_dims(x_data, axis=-1)
v_x_data = np.expand_dims(v_x_data, axis=-1)

#Parameters
filter_num = 64 #Change
batch_size = 256
epoch = 300
data_augmentation = True

# Model
model = Sequential()

# Conv Block 1
model.add(Convolution2D(filter_num, kernel_size=(3, 3), strides=(1,1), padding='same', activation='relu', input_shape=x_data.shape[1:]))
model.add(BatchNormalization())
model.add(Convolution2D(filter_num, kernel_size=(3, 3), strides=(1,1), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
model.add(Dropout(0.2))

# Conv Block 2
model.add(Convolution2D(filter_num*2, kernel_size=(3, 3), strides=(1,1), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Convolution2D(filter_num*2, kernel_size=(3, 3), strides=(1,1), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
model.add(Dropout(0.2))

# Conv Block 3
model.add(Convolution2D(filter_num*4, kernel_size=(3, 3), strides=(1,1), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Convolution2D(filter_num*4, kernel_size=(3, 3), strides=(1,1), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
model.add(Dropout(0.2))

# Conv Block 4
model.add(Convolution2D(filter_num*8, kernel_size=(3, 3), strides=(1,1), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Convolution2D(filter_num*8, kernel_size=(3, 3), strides=(1,1), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
model.add(Dropout(0.2))

# FC Layers
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
#model.add(BatchNormalization())
model.add(Dense(7, activation='softmax'))

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#sgd = SGD(lr=0.001, decay=5*(1e-4), momentum=0.9)
model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

if (data_augmentation):
    datagen = ImageDataGenerator(rotation_range=20 ,width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
    datagen.fit(x_data)

    checkpointer = ModelCheckpoint(filepath='model_13.h5', monitor='val_acc', verbose=0, save_best_only=True, mode='auto')
    model.fit_generator(datagen.flow(x_data, y_data, batch_size=batch_size), steps_per_epoch=row/batch_size, verbose=1, callbacks=[checkpointer], validation_data=(v_x_data, v_y_data), epochs=epoch)
else :
    
    checkpointer = ModelCheckpoint(filepath='model_13.h5', monitor='val_acc', verbose=0, save_best_only=True, mode='auto')
    model.fit(x_data, y_data, batch_size=batch_size, epochs=epoch, callbacks=[checkpointer], validation_data=(v_x_data, v_y_data), shuffle=True)

#model.save('model_4.h5')
