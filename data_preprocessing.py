# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 17:45:28 2018

@author: huang
"""
import sys
import  csv
import time
import collections
import numpy as np
import pandas as pd
import traceback
from sklearn.model_selection import train_test_split
import lightgbm as lgb 

import logging
sys.path.append('..')
from db.conn_db import db,cursor,engine,truncate_table,data_from_mysql,dev_id_train

#sys.path.append('D:/GitHub/jhye_tool')
from flags import FLAGS, unparsed


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.DEBUG)





def logloss(_y, y, weight=None):
#    _y=pred[:,1]
    if weight is None:
        weight = np.ones(_y.size)
    
    _y = list(map(lambda x:np.maximum(1e-7, np.minimum(1 - 1e-7, x)),_y))
    labels=np.zeros(22)
    labels[y]=1
    
    return 'logloss',- np.sum(weight * (labels * np.log(_y))) / np.sum(weight)

def gdbt_data_get_train():
    
    deviceid_train=dev_id_train()
    
    deviceid_packages_02 = pd.read_csv(FLAGS.file_path +'02_deviceid_packages.csv',)
    deviceid_packages_03 = pd.read_csv(FLAGS.file_path +'03_deviceid_packages.csv',)
#    deviceid_packages_04 = pd.read_csv(FLAGS.file_path +'04_deviceid_train.csv',)
    deviceid_packages_05= pd.read_csv(FLAGS.file_path +'05_deviceid_packages.csv',)

    deviceid_train=pd.merge(deviceid_train,deviceid_packages_02,on=['device_id'],how='left') 
    deviceid_train=pd.merge(deviceid_train,deviceid_packages_03,on=['device_id'],how='left') 
    deviceid_train=pd.merge(deviceid_train,deviceid_packages_05,on=['device_id'],how='left') 
    logging.debug(deviceid_train.columns)
    logging.debug(deviceid_train.shape)



    try:
        deviceid_train.drop('device_id', axis=1,inplace = True)
        deviceid_train.drop('sex', axis=1,inplace = True)
        deviceid_train.drop('age', axis=1,inplace = True)
        
        deviceid_train.drop('t1_code', axis=1,inplace = True)
        deviceid_train.drop('t2_code', axis=1,inplace = True)
        deviceid_train.drop('add_list_x', axis=1,inplace = True)
        deviceid_train.drop('add_list_y', axis=1,inplace = True)
    except:
        error_msg = traceback.format_exc()
        print(error_msg)
    logging.debug(deviceid_train.columns)
    logging.debug(deviceid_train.shape)
    logging.debug(deviceid_train.head(2))
    logging.debug(deviceid_train['n_class'].unique())
    return deviceid_train


def gdbt_data_get_test():
    
    deviceid_test=pd.read_csv(FLAGS.file_path+'deviceid_test.csv')
    
    deviceid_packages_02 = pd.read_csv(FLAGS.file_path +'02_deviceid_packages.csv',)
    deviceid_packages_03 = pd.read_csv(FLAGS.file_path +'03_deviceid_packages.csv',)
#    deviceid_packages_04 = pd.read_csv(FLAGS.file_path +'04_deviceid_train.csv',)
    deviceid_packages_05= pd.read_csv(FLAGS.file_path +'05_deviceid_packages.csv',)

    deviceid_test=pd.merge(deviceid_test,deviceid_packages_02,on=['device_id'],how='left') 
    deviceid_test=pd.merge(deviceid_test,deviceid_packages_03,on=['device_id'],how='left') 
    deviceid_test=pd.merge(deviceid_test,deviceid_packages_05,on=['device_id'],how='left') 
    logging.debug(deviceid_test.columns)
    logging.debug(deviceid_test.shape)


    try:
        deviceid_test.drop('device_id', axis=1,inplace = True)
        
        
        deviceid_test.drop('t1_code', axis=1,inplace = True)
        deviceid_test.drop('t2_code', axis=1,inplace = True)
        deviceid_test.drop('add_list_x', axis=1,inplace = True)
        deviceid_test.drop('add_list_y', axis=1,inplace = True)
    except:
        error_msg = traceback.format_exc()
        print(error_msg)
    logging.debug(deviceid_test.columns)
    logging.debug(deviceid_test.shape)
    logging.debug(deviceid_test.head(2))
#    logging.debug(deviceid_test['n_class'].unique())
    return deviceid_test


def lightgbm_data_get_train():
    deviceid_train=dev_id_train()
    
    deviceid_packages_02 = pd.read_csv(FLAGS.file_path +'02_deviceid_packages.csv',)
    deviceid_packages_03 = pd.read_csv(FLAGS.file_path +'03_deviceid_packages.csv',)
#    deviceid_packages_04 = pd.read_csv(FLAGS.file_path +'04_deviceid_train.csv',)
    deviceid_packages_05= pd.read_csv(FLAGS.file_path +'05_deviceid_packages.csv',)

    deviceid_train=pd.merge(deviceid_train,deviceid_packages_02,on=['device_id'],how='left') 
    deviceid_train=pd.merge(deviceid_train,deviceid_packages_03,on=['device_id'],how='left') 
    deviceid_train=pd.merge(deviceid_train,deviceid_packages_05,on=['device_id'],how='left') 
    logging.debug(deviceid_train.columns)
    logging.debug(deviceid_train.shape)



    try:
        deviceid_train.drop('device_id', axis=1,inplace = True)
        deviceid_train.drop('sex', axis=1,inplace = True)
        deviceid_train.drop('age', axis=1,inplace = True)
        
        deviceid_train.drop('t1_code', axis=1,inplace = True)
        deviceid_train.drop('t2_code', axis=1,inplace = True)
        deviceid_train.drop('add_list_x', axis=1,inplace = True)
        deviceid_train.drop('add_list_y', axis=1,inplace = True)
    except:
        error_msg = traceback.format_exc()
        print(error_msg)

    
    print(deviceid_train.shape)
    y_train = deviceid_train['n_class']
    deviceid_train.drop('n_class',axis=1,inplace=True)
    X_train = deviceid_train
    
    
    X_train_part, X_val, y_train_part, y_val = train_test_split(X_train, y_train, train_size = 0.8,random_state = 0)
    logging.debug(X_train_part.head(1))
    logging.debug(y_train_part.head(1))
    ### 数据转换
    lgb_train = lgb.Dataset(X_train_part, y_train_part, free_raw_data=False)
    lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train,free_raw_data=False)

    return lgb_train,lgb_eval,X_val,y_val

def lightgbm_data_get_test():

    deviceid_test=pd.read_csv(FLAGS.file_path+'deviceid_test.csv')
    
    deviceid_packages_02 = pd.read_csv(FLAGS.file_path +'02_deviceid_packages.csv',)
    deviceid_packages_03 = pd.read_csv(FLAGS.file_path +'03_deviceid_packages.csv',)
#    deviceid_packages_04 = pd.read_csv(FLAGS.file_path +'04_deviceid_train.csv',)
    deviceid_packages_05= pd.read_csv(FLAGS.file_path +'05_deviceid_packages.csv',)

    deviceid_test=pd.merge(deviceid_test,deviceid_packages_02,on=['device_id'],how='left') 
    deviceid_test=pd.merge(deviceid_test,deviceid_packages_03,on=['device_id'],how='left') 
    deviceid_test=pd.merge(deviceid_test,deviceid_packages_05,on=['device_id'],how='left') 
    logging.debug(deviceid_test.columns)
    logging.debug(deviceid_test.shape)


    try:
        deviceid_test.drop('device_id', axis=1,inplace = True)
        
        
        deviceid_test.drop('t1_code', axis=1,inplace = True)
        deviceid_test.drop('t2_code', axis=1,inplace = True)
        deviceid_test.drop('add_list_x', axis=1,inplace = True)
        deviceid_test.drop('add_list_y', axis=1,inplace = True)
    except:
        error_msg = traceback.format_exc()
        print(error_msg)
    


    return deviceid_test

gdbt_data_get_train()

