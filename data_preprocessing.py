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
from sklearn.utils import shuffle
import lightgbm as lgb 
from joblib import dump, load, Parallel, delayed


# 数据标准化
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.decomposition import PCA


import logging
sys.path.append('..')
from db.conn_db import db,cursor,engine,truncate_table,data_from_mysql,dev_id_train

#sys.path.append('D:/GitHub/jhye_tool')
from flags import FLAGS, unparsed


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.DEBUG)




#columns=['brand_cnt','type_no_cnt','app_id_weight','app_id_weight','app_t1_weight','app_t2_weight'] \
#+['app_lda_t2_1','app_lda_t2_2','app_lda_t2_3','app_lda_t2_4','app_lda_t2_5',] \
#+['dev_brand_weight','dev_type_no_weight'] \
#+['times_len']




def logloss(_y, y, weight=None):
#    _y=pred[:,1]
    if weight is None:
        weight = np.ones(_y.size)
    
    _y = list(map(lambda x:np.maximum(1e-7, np.minimum(1 - 1e-7, x)),_y))
    labels=np.zeros(22)
    labels[y]=1
    
    return 'logloss',- np.sum(weight * (labels * np.log(_y))) / np.sum(weight)

def test_score(_y, y):
    # y=[1,2,3]
    label=[]
#    _y.drop('DeviceID', axis=1,inplace = True)
    for i in [1,2]:
        for j in range(11):
            label.append(str(i)+'-'+str(j))
    a1 = np.zeros(_y.shape) 
    n=_y.shape[0]
    label_pd=pd.DataFrame(a1,columns=label)
    
    label_map=pd.read_csv(FLAGS.file_path+'label.csv').to_dict(orient='records')[0]
    
    label_map={v: k for k, v in label_map.items()}
    
    y=y.apply(lambda x:label_map[x])
    
    for col in label:
#        logging.debug(y)
        filte=(y.values==col)
#        logging.debug(filte)
        label_pd.ix[filte,col]=1
    
#    _y = list(map(lambda x:np.maximum(1e-7, np.minimum(1 - 1e-7, x)),_y))
    labels=label_pd.values
    
    return - np.sum(  np.sum(labels * np.log(_y)))/n

def data_augmentation(deviceid_train):
    all_class=deviceid_train['n_class'].value_counts( sort=True,).tolist()
    
    max_line=all_class[0]
    logging.debug(all_class[1:])
    for x in deviceid_train.n_class.unique().tolist():
        filte=deviceid_train.n_class.values==x
        tmp_deviceid_train=deviceid_train.ix[filte,:]
        logging.debug(deviceid_train.n_class.values)
        n=max_line-tmp_deviceid_train.shape[0]
        while tmp_deviceid_train.shape[0]<n:
            tmp_deviceid_train=pd.concat([tmp_deviceid_train,tmp_deviceid_train])
        

        deviceid_train=pd.concat([deviceid_train,tmp_deviceid_train[:n]])
    logging.debug(deviceid_train.shape)
    logging.debug(deviceid_train['n_class'].value_counts( sort=True,))
    deviceid_train=shuffle(deviceid_train,random_state =0)
    return deviceid_train

def get_train_data(flag='train'):
    if flag=='train':
        deviceid_train=pd.read_csv(FLAGS.file_path+'deviceid_train.csv')
    elif flag=='test':
        deviceid_train=pd.read_csv(FLAGS.file_path+'deviceid_test.csv')
    '''
    brand 编码
    type_no 编码
    brand_cnt brand 出现的次数
    type_no_cnt type_no 出现的次数
    type_no_w  type_no 在训练数据中 sex为某以类别 出现的在总类别中占比
    type_no2_w  type_no 在训练数据中 age为某以类别 出现的在总类别中占比
    type_no3_w  type_no 在训练数据中 n_class为某以类别 出现的在总类别中占比
    '''
    deviceid_packages_01 = pd.read_csv(FLAGS.file_path +'01_deviceid_packages.csv',)
    
    '''
    'app_id_weight','app_len_t1_43', 'app_len_t2_132', 'app_len_t1_36', 
    'app_len_t2_94','app_len_t2_251', 'app_len_t1_33', 'app_len_t2_124', 'app_len_t1_32', 
    'app_len_t2_223', 'app_len_t2_83','app_len_t2_106', 'app_len_t2_61', 'app_len_t2_159',
    'app_len_t1_17', 'app_len_t2_158'
    app_id_weight app的逆文档频率
    app_len_t1 编码
    app_len_t2 brand 出现的次数
    
    '''
    deviceid_packages_02 = pd.read_csv(FLAGS.file_path +'02_deviceid_packages.csv',)
    
    
    deviceid_packages_03 = pd.read_csv(FLAGS.file_path +'03_deviceid_packages.csv',)
#    deviceid_packages_05= pd.read_csv(FLAGS.file_path +'05_deviceid_packages.csv',)
#    deviceid_packages_06= pd.read_csv(FLAGS.file_path +'06_deviceid_packages.csv',)
#    deviceid_packages_04= pd.read_csv(FLAGS.file_path +'04_deviceid_packages.csv',)
#    deviceid_packages_09= pd.read_csv(FLAGS.file_path +'09_deviceid_packages.csv',)

    deviceid_train=pd.merge(deviceid_train,deviceid_packages_01,on=['device_id'],how='left') 
    deviceid_train=pd.merge(deviceid_train,deviceid_packages_02,on=['device_id'],how='left') 
    deviceid_train=pd.merge(deviceid_train,deviceid_packages_03,on=['device_id'],how='left')
#    deviceid_train=pd.merge(deviceid_train,deviceid_packages_05,on=['device_id'],how='left')
#    deviceid_train=pd.merge(deviceid_train,deviceid_packages_06,on=['device_id'],how='left') 
#    deviceid_train=pd.merge(deviceid_train,deviceid_packages_04,on=['device_id'],how='left')
#    deviceid_train=pd.merge(deviceid_train,deviceid_packages_09,on=['device_id'],how='left')
    deviceid_train=deviceid_train.fillna(0)
    return deviceid_train
# flag  in ['all','sex','age','device_id']
def gdbt_data_get_train(flag='all'):
    
    deviceid_train=get_train_data('train')
    logging.debug(deviceid_train.columns)
    logging.debug(deviceid_train.shape)


#    deviceid_train=data_augmentation(deviceid_train)
    try:
        if flag!='device_id':
            deviceid_train.drop('device_id', axis=1,inplace = True)
        if flag!='sex' and flag!='age':
            deviceid_train.drop('sex', axis=1,inplace = True)
        if flag!='age':
            deviceid_train.drop('age', axis=1,inplace = True)
        if flag!='n_class':
            deviceid_train.drop('n_class', axis=1,inplace = True)
        
#        deviceid_train.drop('t1_code', axis=1,inplace = True)
#        deviceid_train.drop('t2_code', axis=1,inplace = True)
#        deviceid_train.drop('add_list', axis=1,inplace = True)

    except:
        error_msg = traceback.format_exc()
        print(error_msg)
    
    logging.debug(deviceid_train.columns)
    logging.debug(deviceid_train.shape)
    logging.debug(deviceid_train.head(2))
    
    return deviceid_train


def gdbt_data_get_eval(flag='all'):
    deviceid_train=get_train_data('train')
    logging.debug(deviceid_train.columns)
    logging.debug(deviceid_train.shape)
    try:
        if flag!='device_id':
            deviceid_train.drop('device_id', axis=1,inplace = True)
        if flag!='sex' and flag!='age':
            deviceid_train.drop('sex', axis=1,inplace = True)
        if flag!='age':
            deviceid_train.drop('age', axis=1,inplace = True)
        if flag!='n_class':
            deviceid_train.drop('n_class', axis=1,inplace = True)
#        deviceid_train.drop('add_list', axis=1,inplace = True)

    except:
        error_msg = traceback.format_exc()
        print(error_msg)
#    deviceid_train=data_augmentation(deviceid_train)
    logging.debug(deviceid_train.columns)
    logging.debug(deviceid_train.shape)
    logging.debug(deviceid_train.head(2))
    deviceid_train=deviceid_train.sample(frac=0.3).reindex()
    
    return deviceid_train

def gdbt_data_get_test(flag='all'):
    
    deviceid_test=get_train_data('test')
    
    logging.debug(deviceid_test.columns)
    logging.debug(deviceid_test.shape)


    try:
        deviceid_test.drop('device_id', axis=1,inplace = True)
        
        
#        deviceid_test.drop('t1_code', axis=1,inplace = True)
#        deviceid_test.drop('t2_code', axis=1,inplace = True)
#        deviceid_test.drop('add_list', axis=1,inplace = True)
    except:
        error_msg = traceback.format_exc()
        print(error_msg)
    logging.debug(deviceid_test.columns)
    logging.debug(deviceid_test.shape)
    logging.debug(deviceid_test.head(2))
#    logging.debug(deviceid_test['n_class'].unique())
    return deviceid_test


def lightgbm_data_get_train():
    
    deviceid_train=get_train_data('train')
    logging.debug(deviceid_train.columns)
    logging.debug(deviceid_train.shape)


#    deviceid_train=data_augmentation(deviceid_train)
    try:
            deviceid_train.drop('device_id', axis=1,inplace = True)
            deviceid_train.drop('sex', axis=1,inplace = True)
            deviceid_train.drop('age', axis=1,inplace = True)
#            deviceid_train.drop('n_class', axis=1,inplace = True)
        
#        deviceid_train.drop('t1_code', axis=1,inplace = True)
#        deviceid_train.drop('t2_code', axis=1,inplace = True)
#        deviceid_train.drop('add_list', axis=1,inplace = True)

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

    deviceid_test=get_train_data('test')
    logging.debug(deviceid_test.columns)
    logging.debug(deviceid_test.shape)


    try:
        deviceid_test.drop('device_id', axis=1,inplace = True)
        
        
#        deviceid_test.drop('t1_code', axis=1,inplace = True)
#        deviceid_test.drop('t2_code', axis=1,inplace = True)
#        deviceid_test.drop('add_list', axis=1,inplace = True)
    except:
        error_msg = traceback.format_exc()
        print(error_msg)
    


    return deviceid_test


def cnn_read_data():
    # 分别初始化对特征和目标值的标准化器
    ss_X = StandardScaler()    
    
    
    deviceid_train=dev_id_train()
    deviceid_test=pd.read_csv(FLAGS.file_path+'deviceid_test.csv')
    
#    return
#    deviceid_packages_01 = pd.read_csv(FLAGS.file_path +'01_deviceid_packages.csv',)
    deviceid_packages_02 = pd.read_csv(FLAGS.file_path +'02_deviceid_packages.csv',)
#    deviceid_packages_03 = pd.read_csv(FLAGS.file_path +'03_deviceid_packages.csv',)
#    deviceid_packages_04 = pd.read_csv(FLAGS.file_path +'04_deviceid_packages.csv',)
    deviceid_packages_05= pd.read_csv(FLAGS.file_path +'05_deviceid_packages.csv',)
    deviceid_packages_06= pd.read_csv(FLAGS.file_path +'06_deviceid_packages.csv',)

#    deviceid_packages=pd.merge(deviceid_packages_02,deviceid_packages_03,on=['device_id'],how='left') 
    deviceid_packages=pd.merge(deviceid_packages_02,deviceid_packages_05,on=['device_id'],how='left') 
    deviceid_packages=pd.merge(deviceid_packages,deviceid_packages_06,on=['device_id'],how='left') 
    logging.debug(deviceid_packages.columns)
    logging.debug(deviceid_packages.shape)
    
    
    deviceid_train=pd.merge(deviceid_train,deviceid_packages,on=['device_id'],how='left') 
    deviceid_test=pd.merge(deviceid_test,deviceid_packages,on=['device_id'],how='left') 
    y_train=deviceid_train['n_class']
    deviceid_train.drop('n_class', axis=1,inplace = True)

    try:        
        deviceid_train.drop('add_list', axis=1,inplace = True)
        deviceid_train.drop('device_id', axis=1,inplace = True)
        deviceid_train.drop('sex', axis=1,inplace = True)
        deviceid_train.drop('age', axis=1,inplace = True)
        deviceid_test.drop('device_id', axis=1,inplace = True)

    except:
        error_msg = traceback.format_exc()
        print(error_msg)
        
    
    deviceid_train = ss_X.fit_transform(deviceid_train)
    deviceid_test = ss_X.fit_transform(deviceid_test)


    
    logging.debug(deviceid_train.shape)
    logging.debug(y_train.shape)
    logging.debug(deviceid_test.shape)
    
    return deviceid_train,y_train,deviceid_test
    # 分别对训练和测试数据的特征以及目标值进行标准化处理

def get_all_data_kmeans():
#    return
#    deviceid_packages_01 = pd.read_csv(FLAGS.file_path +'01_deviceid_packages.csv',)
    deviceid_packages_02 = pd.read_csv(FLAGS.file_path +'02_deviceid_packages.csv',)
#    deviceid_packages_03 = pd.read_csv(FLAGS.file_path +'03_deviceid_packages.csv',)
#    deviceid_packages_04 = pd.read_csv(FLAGS.file_path +'04_deviceid_packages.csv',)
    deviceid_packages_05= pd.read_csv(FLAGS.file_path +'05_deviceid_packages.csv',)
    deviceid_packages_06= pd.read_csv(FLAGS.file_path +'06_deviceid_packages.csv',)

#    deviceid_packages=pd.merge(deviceid_packages_02,deviceid_packages_03,on=['device_id'],how='left') 
    deviceid_packages=pd.merge(deviceid_packages_02,deviceid_packages_05,on=['device_id'],how='left') 
    deviceid_packages=pd.merge(deviceid_packages,deviceid_packages_06,on=['device_id'],how='left') 
    logging.debug(deviceid_packages.columns)
    logging.debug(deviceid_packages.shape)
    
    try:        
        deviceid_packages.drop('add_list', axis=1,inplace = True)
        deviceid_packages.drop('device_id', axis=1,inplace = True)

    except:
        error_msg = traceback.format_exc()
        print(error_msg)
    deviceid_packages=deviceid_packages.fillna(0)
    return deviceid_packages
    
def data_normalization(train):
    try:
        min_max_scaler = load(FLAGS.tmp_data_path+'MinMaxScaler_model.joblib_dat')
    except:
        min_max_scaler = MinMaxScaler() 
        min_max_scaler.fit(train)
    train = min_max_scaler.transform(train)  
    ret=dump(min_max_scaler, FLAGS.tmp_data_path+'MinMaxScaler_model.joblib_dat')
    return train
    
def data_pca(train):
    try:
        
        pca = load(FLAGS.tmp_data_path+'PCA_model.joblib_dat')
    except:
        pca = PCA(n_components=FLAGS.pca_rate)
        pca.fit(train)
    train = pca.transform(train)  
    logging.debug("pca后维度:"+str(train.shape))
    ret=dump(pca, FLAGS.tmp_data_path+'PCA_model.joblib_dat')
    return train

def remove(file):
    import os
    if  os.path.exists(file):
        os.remove(file)
if __name__ == "__main__":
    gdbt_data_get_train()

