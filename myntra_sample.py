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














