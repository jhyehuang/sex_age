# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 11:42:17 2018

@author: admin
"""
import sys
import time
import pandas as pd
import numpy as np
import os

sys.path.append('..')

file_path='../data/sex_age'
files=os.listdir(file_path)
file_names=[]
for file in files:
    file_name=file_path+'/'+file
    file_names.append(file_name)
print(file_names)

data_src=pd.read_csv(file_name,nrows=200,sep='\t')
print(data_src)
