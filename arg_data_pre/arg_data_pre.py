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

file_path='../../data/sex_age/'
files=os.listdir(file_path)
file_names=[]
for file in files:
    file_name=file_path+'/'+file
    file_names.append(file_name)
print(file_names)

all_list=[]
with open(file_path+'deviceid_packages.tsv','r') as fh:
    for line in fh: 
        
        line_list=line.replace('\t',',').replace(' ',',').replace('\n','').split(',')
        print(line_list)
        id_dict={}
        id_no=line_list[0]
        app_list=line_list[1:]
        id_dict['id']=id_no
        for i,x in enumerate(line_list):
            id_dict['app_'+str(i)]=x
        
        all_list.append(id_dict)
       
data_src=pd.DataFrame(x for x in all_list)
#data_src=pd.read_csv(file_path+'package_label.csv')

print(data_src.head(5))
data_src.to_csv(file_path+'deviceid_packages.csv')




