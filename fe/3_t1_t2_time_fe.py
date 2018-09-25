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
import json

'''
# 初始化数据库连接，使用pymysql模块 # MySQL的用户：root, 
密码:root, 端口：3306,
数据库：
mydb 
'''

sys.path.append('..')
from db.conn_db import db,cursor,engine,truncate_table,data_from_mysql,get_package_dict
from flags import FLAGS, unparsed
from functools import reduce
import logging

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.DEBUG)



file_path=FLAGS.file_path

def deviceid_package_start_close_train(deviceid,app_id='0'):
    sql='select * from deviceid_package_start_close where device_id=\"'
    sql=sql+deviceid+'\" '
    sql=sql+'and app_id=\"'
    sql=sql+app_id+'\"'
#    print(sql)
    ret=data_from_mysql(sql)
    return ret


def get_times(dev_id,app_id_list):
    ret=deviceid_package_start_close_train(dev_id,app_id_list)
#        print(ret['close'].map(int),ret['start'].map(int))
    if ret.shape[0]<1:
        return 0
    ret['close'] = ret['close'].map(int)
    ret['start'] = ret['start'].map(int)
#        print(ret)
    condition = sum(ret['close'].map(int).values-ret['start'].map(int).values)/1000/60/60
    return int(condition)

def get_times_len(dev_id,app_id_list):
    ret=0
    for app_id in app_id_list:
        ret=ret+get_times(dev_id,app_id)
    print(dev_id,ret)
    return int(ret)
    
def dev_id_train():
    sql='select * from deviceid_train'
    deviceid_train=data_from_mysql(sql)
    return deviceid_train


def app_get_t1(dev_id,app_list):
    logging.debug(app_list)
    if len(app_list)<1:
        return {}
    tx_list=[]
    for app_id in app_list:
        t1_dict={}
        t2=get_package_dict(app_id,'t1,t2')
        
        if len(t2)<1:
#            print(t2)
            continue
        t2=t2[0]
        t1_dict['t1']=t2.get('t1','0')
        t1_dict['t2']=t2.get('t2','0')
        t1_dict['app_id']=app_id
        tx_list.append(t1_dict)
    tx_pd=pd.DataFrame(x for x in tx_list)
    if tx_pd.shape[0]<1:
        return {}
    
    result_t1={}
    for t1 in tx_pd.t1.unique():
        total_times=get_times_len(dev_id,tx_pd.ix[tx_pd.t1.values==t1,'app_id'].tolist())
        result_t1[t1]=total_times
    logging.debug(result_t1)
    return result_t1


def app_get_t2(dev_id,app_list):
    logging.debug(app_list)
    if len(app_list)<1:
        return {}
    tx_list=[]
    for app_id in app_list:
        t1_dict={}
        t2=get_package_dict(app_id,'t1,t2')
        if len(t2)<1:
#            print(t2)
            continue
        t2=t2[0]
        t1_dict['t1']=t2.get('t1','0')
        t1_dict['t2']=t2.get('t2','0')
        t1_dict['app_id']=app_id
        tx_list.append(t1_dict)
    tx_pd=pd.DataFrame(x for x in tx_list)
    if tx_pd.shape[0]<1:
        return {}
#    print(tx_pd)

    result_t2={}
    for t2 in tx_pd.t2.unique():
        total_times=get_times_len(dev_id,tx_pd.ix[tx_pd.t2.values==t2,'app_id'].tolist())
        result_t2[t2]=total_times
    return result_t2

def devid_app_times_tx(deviceid_packages,package_label):
    
    def app_list(text):
        app_list=text.split('|')
#        print (app_list)
        return app_list
    deviceid_packages['app_list']=deviceid_packages['app_id_list'].apply(lambda line:app_list(line)).tolist()
    # 使用 t1 app 的时长
    # 使用 t2 app 的时长
    deviceid_packages['t1_app_time']=deviceid_packages.apply(lambda line:app_get_t1(line['device_id'],line['app_list']),axis=1)
    deviceid_packages['t2_app_time']=deviceid_packages.apply(lambda line:app_get_t2(line['device_id'],line['app_list']),axis=1)
    
    columns=[]
#    logging.debug(FLAGS.t1_feature.replace('\'','').split(','))
#    for x in FLAGS.t1_feature.replace('\'','').split(','):
#        columns.append('app_time_t1_'+str(x))
#    for x in FLAGS.t2_feature.replace('\'','').split(','):
#        columns.append('app_time_t2_'+str(x))
        
    for x in package_label['t1'].unique():
        deviceid_packages['app_time_t1_'+str(x)]=int(0)
        columns.append('app_time_t1_'+str(x))

    for x in package_label['t2'].unique():
        deviceid_packages['app_time_t2_'+str(x)]=int(0)
        columns.append('app_time_t2_'+str(x))
    
    for x in package_label['t1'].unique():
        _x=[]
        for i in range(deviceid_packages.shape[0]):
            _x.append(str(x))
#        print(_x)
        _x=pd.DataFrame({'a':_x},dtype='category')

        def c(a,b):
#            print(a,b)
            ert=(str(a) in b.keys())
#            print(ert)
            return ert
        a=list(map(c,_x['a'],deviceid_packages['t1_app_time']))


        filte=np.logical_and(a,True)
        def get_values(t1_dict):
            return t1_dict[str(x)]/np.sum(list(t1_dict.values()))
            
        values=deviceid_packages.ix[filte,'t1_app_time'].apply(lambda x:get_values(x))
#        print(filte)
        deviceid_packages.ix[filte,'app_time_t1_'+str(x)]=values
        
        
    for x in package_label['t2'].unique():
        _x=[]
        for i in range(deviceid_packages.shape[0]):
            _x.append(str(x))
#        print(_x)
        _x=pd.DataFrame({'a':_x},dtype='category')

        def c(a,b):
            ert=(str(a) in b.keys())
#            print(ert)
            return ert
        a=list(map(c,_x['a'],deviceid_packages['t2_app_time']))


        filte=np.logical_and(a,True)
            
        values=deviceid_packages.ix[filte,'t2_app_time'].apply(lambda x:get_values(x))
#        print(filte)
        deviceid_packages.ix[filte,'app_time_t2_'+str(x)]=values
    columns.append('device_id')
    logging.debug(columns)
    
    return deviceid_packages.ix[:, columns]




def devid_times(deviceid_packages):

#    print(deviceid_packages.head(5))
    def app_list(text):
        app_list=text.split('|')
#        print (app_list)
        return app_list
    deviceid_packages['app_list']=deviceid_packages['app_id_list'].apply(lambda line:app_list(line)).tolist()

#    print(deviceid_packages.shape)
     
    deviceid_packages['times_len']=deviceid_packages.apply(lambda line:get_times_len(line['device_id'],line['app_list']) ,axis=1)

    return deviceid_packages.ix[:, ['device_id','times_len']]
    
 
    
def compute_date():
    import multiprocessing

    pool = multiprocessing.Pool(processes=2)
    deviceid_packages=pd.read_csv(file_path+'deviceid_packages.csv')
    
    package_label=pd.read_csv(file_path+'package_label.csv')
    package_label['t1']=package_label['t1'].astype('category').values.codes
    package_label['t2']=package_label['t2'].astype('category').values.codes
    result = []
    result.append(pool.apply_async(devid_times, (deviceid_packages, )))
    result.append(pool.apply_async(devid_app_times_tx, (deviceid_packages,package_label, )))
    pool.close()
    pool.join()
        
    deviceid_packages=pd.merge(result[0].get(),result[1].get(),on=['device_id'],how='left') 
    
    print(deviceid_packages.head(5))
    
    deviceid_packages.to_csv(file_path+'03_deviceid_packages.csv',index= False)
    
    
if __name__=='__main__':
    start_time=time.time()

    compute_date()

# id,
    end_time=time.time()
    print('耗时:',end_time-start_time)


