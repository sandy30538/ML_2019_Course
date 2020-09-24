import sys
import numpy as np
import pandas as pd
import csv
import random as rand
import math
from keras.models import load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import BatchNormalization, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam, RMSprop, Adagrad
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint

# File
input_1     = sys.argv[1]           # test.csv
output_1    = sys.argv[2]           # ans.csv

# Parameter
row = 7178          # 7178 Data
input_size = 48     # 48 x 48 pixels

# Open test.csv file
data_x = pd.read_csv(input_1, encoding = 'big5')
np_data_x = np.array(data_x)
np_data_x = np_data_x.tolist()  

# reshape the test data and label_data
test_data = list()
for i in range(row):
    a = np_data_x[i][1].split()
    b = np.array([float(j) for j in a])
    c = b.reshape((48,48))
    c = c.tolist()

    test_data.append(c)

test_data = np.array(test_data)
test_data = np.expand_dims(test_data, axis=-1)

# Compile model
model_13 = load_model('model_13.h5')
model_16 = load_model('model_16.h5')
model_18 = load_model('model_18.h5')
model_20 = load_model('model_20.h5')
model_21 = load_model('model_21.h5')
#model_22 = load_model('model_22.h5')
#model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

output_13 = model_13.predict(test_data, batch_size=128)
ans_13 = np.argmax(output_13, axis=1)
output_16 = model_16.predict(test_data, batch_size=128)
ans_16 = np.argmax(output_16, axis=1)
output_18 = model_18.predict(test_data, batch_size=128)
ans_18 = np.argmax(output_18, axis=1)
output_20 = model_20.predict(test_data, batch_size=128)
ans_20 = np.argmax(output_20, axis=1)
output_21 = model_21.predict(test_data, batch_size=128)
ans_21 = np.argmax(output_21, axis=1)
#output_22 = model_22.predict(test_data, batch_size=128)
#ans_22 = np.argmax(output_22, axis=1)

ans = list()
for i in range(row):
    tmp = [0, 0, 0, 0, 0, 0, 0]
    tmp[ans_13[i]] = tmp[ans_13[i]] + 1;
    tmp[ans_16[i]] = tmp[ans_16[i]] + 1;
    tmp[ans_18[i]] = tmp[ans_18[i]] + 1;
    tmp[ans_20[i]] = tmp[ans_20[i]] + 1;
    tmp[ans_21[i]] = tmp[ans_21[i]] + 1;
    #tmp[ans_22[i]] = tmp[ans_22[i]] + 1;
    
    index = 0
    best = 0
    for k in range(7):
        if (tmp[k] > best):
            index = k
            best = tmp[k]
    ans.append(index)

with open (output_1, 'w') as wr:
    names = ['id', 'label']
    writer = csv.DictWriter(wr, fieldnames=names)
    writer.writeheader()
    for id in range(row) :
        writer.writerow({'id': id, 'label': ans[id]})
