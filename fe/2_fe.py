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
from subprocess import *

'''
# 初始化数据库连接，使用pymysql模块 # MySQL的用户：root, 
密码:root, 端口：3306,
数据库：
mydb 
'''

sys.path.append('..')
from db.conn_db import db,cursor,engine,truncate_table,data_from_mysql
from flags import FLAGS, unparsed
from functools import reduce

file_path=FLAGS.file_path

def dev_id_train():
    sql='select * from deviceid_train'
    deviceid_train=data_from_mysql(sql)
    return deviceid_train




def devid_app_count():
    
    deviceid_packages=pd.read_csv(file_path+'deviceid_packages.csv')
    package_label=pd.read_csv(file_path+'package_label.csv')
    deviceid_train=dev_id_train()
#    print(deviceid_packages.head(5))
    def app_count(text):
        app_list=text.split('|')
        return len(app_list)
    deviceid_packages['app_len']=deviceid_packages['add_id_list'].apply(lambda line:app_count(line))
    package_label['t1']=package_label['t1'].astype('category').values.codes
    package_label['t2']=package_label['t2'].astype('category').values.codes
    def app_list(text):
        app_list=text.split('|')
#        print (app_list)
        return app_list
    app_mtrix=deviceid_packages['add_id_list'].apply(lambda line:app_list(line)).tolist()
#    app_mtrix=app_mtrix['add_id_list']
#    print(package_label['t1'].max())
#    print(package_label['t2'].max())
    
    def get_label_t1_1(l):
        print(l)
        ret=list(map(get_label_2_t1,l))
        condition = lambda t: t != ""
        ret= list(filter(condition, ret))
        ret=list(map(str,ret))
        if len(ret)<1:
            ret.append('0')
        return ''.join(ret)
  
    def get_label_t2_1(l):
        print(l)
        ret=list(map(get_label_2_t2,l))
        condition = lambda t: t != ""
        ret= list(filter(condition, ret))
        ret=list(map(str,ret))
        if len(ret)<1:
            ret.append('0')
        return ''.join(ret)
  
    def get_label_2_t1(l):
        filer=package_label['app_id'].astype('category').values==l
        label=package_label.ix[filer,'t1'].values.tolist()
        if len(label)<1:
            return ''
        return label.pop()
    
    def get_label_2_t2(l):
        filer=package_label['app_id'].astype('category').values==l
        label=package_label.ix[filer,'t2'].values.tolist()
        if len(label)<1:
            return ''
        return label.pop()
    t1_mtrix=list(map(get_label_t1_1,app_mtrix))
    t2_mtrix=list(map(get_label_t2_1,app_mtrix))
    
    
    deviceid_packages['t1_code']=np.array(t1_mtrix).reshape(-1,1)
    deviceid_packages['t2_code']=np.array(t2_mtrix).reshape(-1,1)
    
    deviceid_packages['t1_code']=deviceid_packages['t1_code'].astype('category').values.codes
    deviceid_packages['t2_code']=deviceid_packages['t2_code'].astype('category').values.codes
    
#    deviceid_train=pd.merge(deviceid_train,deviceid_packages,on=['device_id'],how='left') 
    
 
    
#    print(deviceid_train.head(5))
    
    deviceid_packages.to_csv(file_path+'02_deviceid_packages.csv', columns=['device_id','app_len','t1_code','t2_code'],index= False)
    
#    deviceid_test=pd.read_csv(file_path+'deviceid_test.csv')
#    deviceid_test=pd.merge(deviceid_test,deviceid_packages,on=['device_id'],how='left')
#    deviceid_test.to_csv(file_path+'02_deviceid_test.csv', columns=['device_id','app_len','t1_code','t2_code'],index= False)
    
    
    
if __name__=='__main__':
    start_time=time.time()

    devid_app_count()

# id,
    end_time=time.time()
    print('耗时:',end_time-start_time)



