# -*- coding: utf-8 -*-
"""finalYear.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1yj0NINQShvxMIb2tmvAX4OgOqCff5O2g
"""

# from google.colab import drive
# drive.mount('/content/drive')

# with open('/content/drive/My Drive/Lab2/potter.txt', 'w') as f:
#   f.write('Hello Google Drive!')
# !cat /content/drive/My\ Drive/foo.txt

import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import advanced_activations
from keras.layers import Activation
from keras.layers.core import Dense
from keras.layers import LSTM
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.callbacks import ModelCheckpoint
import numpy as np

model2 = Sequential([
                    Dense(512, input_shape=(1200,), activation='relu'),
                    Dense(128,activation = 'relu'),
                    Dense(64, activation = 'relu'),
                    Dense(32, activation = 'relu'),
                    Dense(16, activation='softmax'),
                    Dense(5, activation='sigmoid')
])
model2.summary()

#with open('/content/drive/Shared with me/potter.txt', 'r') as f:
#    f.read('Hello Google Drive!')
rawStr =''
rawData = []
f = open("finalDataB.txt", "r")
rawStr = f.read()
rawData= rawStr.split('\n')
#print(len(rawData))
data = []

# f = open("knee.txt", "r")
# data =[]
# for i in f.read().split(', '):
#   data.append(float(i))
# f.close()
# y1 = data
print("done reading")
dataLabel=[]
for i in range(len(rawData)):
  x = rawData[i].split(", ")
  #get labels
  dataLabel.append(int(x[-1]))
  #get inputs
  data.append([])
  for j in range(len(x)-1):
    x[j] = float(x[j])
    data[i].append(x[j])# = (np.asarray(x[slice(0,-1)]))

print("data separated")
data = np.asarray(data)
print(len(dataLabel), len(data))
print(len(data[0]))
# for i in data:
#   print(i)

adam = Adam(lr=0.001,beta_1=0.9,beta_2=0.999)
chk = ModelCheckpoint('model/bestSquatLinear.h5', monitor='val_acc', save_best_only=True, mode='max', verbose=1)

model2.compile(Adam(lr=0.001,beta_1=0.9,beta_2=0.999), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model2.fit(data, dataLabel, validation_split=0.1, callbacks=[chk], batch_size=128, epochs=3000, shuffle=True, verbose=2)

model2.get_weights()
model2.save('model/squatLinear.h5')