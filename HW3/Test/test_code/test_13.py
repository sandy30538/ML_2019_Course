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
model = load_model('model_13.h5')
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

output = model.predict(test_data, batch_size=128)
ans = np.argmax(output, axis=1)

with open (output_1, 'w') as wr:
    names = ['id', 'label']
    writer = csv.DictWriter(wr, fieldnames=names)
    writer.writeheader()
    for id in range(row) :
        writer.writerow({'id': id, 'label': ans[id]})
