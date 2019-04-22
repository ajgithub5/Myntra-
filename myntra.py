#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 10:35:05 2018

@author: ajay
"""

import pandas as pd
import numpy as np
import os
import re
import csv
import random

os.chdir("/home/ajay/Documents/Myntra")

data = pd.read_csv("items_myntra.csv")
data.head()
data.columns
len(data['Inode_Breadcrumb'].unique().tolist())

data.shape

'''p_data = data
p_data.head()
p_data.shape

p_data["p_name"]=p_data["Name"]
p_data["p_name"].head()

#p_data['p_name'][0]=re.sub('"','',p_data['p_name'][0])

p_data['p_name'][0] = p_data['p_name'][0][1:-1]

#removing all the inverted commas
for i in range(0,len(p_data)):
    p_data['p_name'][i]=p_data['p_name'][i][1:-1]

p_data['p_name'].head()'''

subset = data.sample(10000)
subset.shape
subset.head()
subset.to_csv('myntra.csv')


import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
numpy.random.seed(7)

from keras.datasets import imdb
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = top_words)
embedding_vecor_length = 32
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64)













