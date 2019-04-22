#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 10:21:49 2018

@author: ajay
"""

import os
import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense,Activation,Flatten
from keras.layers import LSTM
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

os.chdir("/home/ajay/Documents/Myntra")
df = pd.read_csv('myntra.csv')

df['newname']=df['Name']
# removing double quotes at the ends and converting to lower case
def doubleq_lower(df):
    for i in range(0,len(df)):
        if(df[i][0]=='"' and df[i][-1]=='"'): 
            df[i]=df[i][1:-1]
            df[i]=df[i].lower()
        elif(df[i][0]=='"' and not df[i][-1]=='"'):
            df[i]=df[i][1:]
            df[i]=df[i].lower()
        elif(df[i][-1]=='"' and not df[i][0]=='"'):
            df[i]=df[i][:-1]
            df[i]=df[i].lower()
        else:
            df[i]=df[i].lower()
            
doubleq_lower(df['newname'])
doubleq_lower(df['Brand'])

# removal of special characters
def spclchar(df):
    df=re.sub(r'[^a-zA-Z0-9]+',' ',df)
    return df

df['newname'] = df['newname'].apply(spclchar)
df['Brand'] = df['Brand'].apply(spclchar)

colors= list(df['Color'])
colors = np.array(colors)
colors = list(np.unique(colors))

# Stopwords of color list
stop_colors = []
for i in range(0,len(colors)):
    if colors[i]!='nan':
        stop_colors.append(colors[i])

#word tokenize the name column
def wordtokens(df):
    df=word_tokenize(df)
    return(df)
df['newname']=df['newname'].apply(wordtokens)

#stopwords brand
df['Brand'] =[word_tokenize(i) for i in df['Brand']]
for i in range(0,len(df)):
    df['newname'][i] =[w for w in df['newname'][i] if not w in df['Brand'][i]]

#remove colors from the newname column
def stopcolors(df):
    df = [w for w in df if not w in stop_colors]
    return(df)
df['newname']=df['newname'].apply(stopcolors)
 
#remove gender
gender = ['men','man','women','woman','male','female','unisex','girls','boys','girl','boy','kid','kids']
def stopgender(df):
    df = [w for w in df if not w in gender]
    return(df)
df['newname']=df['newname'].apply(stopgender)


#stopwords using nltk.stopwords
stop = set(stopwords.words('english'))
for i in range(0,len(df)):
    df['newname'][i] = [w for w in df['newname'][i] if not w in stop]

#df['newname'][0]
#df['newname_string'] = df['newname'].apply(lambda x: ' '.join(x))

#creating a unique Bag of word for preparing vocab dictionary 
BOW = []
for i in range(0,len(df)):
    for j in range(0,len(df['newname'][i])):
        BOW.append(df['newname'][i][j])
        
unique_BOW= set(BOW)  #unique list of words in Name column
len(unique_BOW)
len(BOW)

# preparing vocab dictionary
dict_name = {ch:i for i, ch in enumerate(unique_BOW)}
dict_name_rev = {i:ch for i,ch in enumerate(unique_BOW)}

#preparing number sequence according to preprocessed product name
df['name2num']=df['newname']
for i in range(0,len(df)):
    sequence_num = []
    for j in range(0,len(df['newname'][i])):
        num = dict_name[df['newname'][i][j]]
        sequence_num.append(num)
    df['name2num'][i] = sequence_num
    

#Encoding inode breadcrumb
unique_inode = list(set(df['Inode_Breadcrumb']))
dict_inode = {ch:i for i,ch in enumerate(unique_inode)}
dict_inode_rev = {i:ch for i,ch in enumerate(unique_inode)}

#len(unique_inode)

#dict_inode[df['Inode_Breadcrumb'][1]]

# Column containing the vector of relevant inode breadcrumb
df['inode_ix']= df['Inode_Breadcrumb'].apply(lambda ch:dict_inode[ch])

name2num = []
for i in range(0,len(df['name2num'])):
    name2num = name2num.append(df['name2num'][i])


#padding
max_len = 10
df['name2num'] = df['name2num'].apply(lambda x: pad_sequences([x],maxlen=max_len,padding='pre',truncating='pre'))
df['name2num'] = df['name2num'].apply(lambda x:x[0])

print(df['name2num'].head())

#split into test and train
vocab_size = len(unique_BOW)
x_train,x_test,y_train,y_test = train_test_split(df['name2num'],df['inode_ix'],test_size=0.3,random_state = 123)

#embedding_vecor_length = 32
 #   model = Sequential()
  #  model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
#    model.add(LSTM(100))
 #   model.add(Dense(1, activation='relu'))
  #  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
   # print(model.summary())''
x_train = x_train.reset_index(drop = True)
x_train_list= []
for i in range(len(x_train)):
      x_train_list.append(x_train[i])
x_train_list
print(len(x_train))   

model = Sequential()
model.add(Embedding(vocab_size,32,input_length = max_len))
model.add(LSTM(100))
model.add(Dense(232,activation='softmax'))
model.compile(optimizer = 'adam',
              loss ='sparse_categorical_crossentropy',
              metrics = ['accuracy'])
print(model.summary())

model.fit(x_train,y_train,batch_size=100,epochs=10,validation_data=(x_test,y_test))





        
