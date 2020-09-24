import sys
import numpy as np
import pandas as pd
import csv
import random as rand
import math
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import BatchNormalization, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam, RMSprop, Adagrad
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
#from keras.regularizers import l1,l2,l1l2

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

print(np.shape(x_data))
print(np.shape(y_data))
print(np.shape(v_x_data))
print(np.shape(v_y_data))

#Parameters
filter_num = 20
batch_size = 128
epoch = 40
data_augmentation = False


# Model
model = Sequential()

model.add(Convolution2D(filter_num, kernel_size=(3, 3), strides=(1,1), padding='same', input_shape=x_data.shape[1:]))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Convolution2D(filter_num, kernel_size=(3,3), strides=(1,1)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))

model.add(Convolution2D(filter_num*2, kernel_size=(3, 3), strides=(1,1), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Convolution2D(filter_num*2, kernel_size=(3,3), strides=(1,1)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))

model.add(Convolution2D(filter_num*4, kernel_size=(3, 3), strides=(1,1), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Convolution2D(filter_num*4, kernel_size=(3,3), strides=(1,1)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(7))
model.add(Activation('softmax'))

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

if (data_augmentation):

    datagen = ImageDataGenerator(rotation_range=20 ,width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    datagen.fit(train_data)

    model.fit_generator(datagen.flow(train_data, label_data, batch_size=batch_size), steps_per_epoch=row/batch_size, verbose=1, epochs=epoch, max_queue_size=40, workers=12, use_multiprocessing=True)
else :
    
    checkpointer = ModelCheckpoint(filepath='model_4.h5', monitor='val_acc', verbose=0, save_best_only=True, mode='auto')
    model.fit(x_data, y_data, batch_size=batch_size, epochs=epoch, callbacks=[checkpointer], validation_data=(v_x_data, v_y_data), shuffle=True)

#model.save('model_4.h5')