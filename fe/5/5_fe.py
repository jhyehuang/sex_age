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
import logging

'''
# 初始化数据库连接，使用pymysql模块 # MySQL的用户：root, 
密码:root, 端口：3306,
数据库：
mydb 
'''

sys.path.append('..')
sys.path.append('../..')
from db.conn_db import db,cursor,engine,truncate_table,data_from_mysql,dev_id_train,get_package_label
from flags import FLAGS, unparsed
from functools import reduce

import time, datetime

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.DEBUG)

file_path=FLAGS.file_path


def time_to_hour(timeStamp):
    # 字符类型的时间
    timeArray = time.localtime(timeStamp)
#    print(timeArray)
    # 转为时间数组
#    timeArray = time.strptime(timeArray, "%Y-%m-%d %H:%M:%S")
    return timeArray.tm_hour   # 2013

def time_to_day(timeStamp):
    # 字符类型的时间
    timeArray = time.localtime(timeStamp)
    # 转为时间数组
#    timeArray = time.strptime(timeArray, "%Y-%m-%d %H:%M:%S")
    return timeArray.tm_mday   # 2013

def time_to_mon(timeStamp):
    # 字符类型的时间
    timeArray = time.localtime(timeStamp)
    # 转为时间数组
#    timeArray = time.strptime(timeArray, "%Y-%m-%d %H:%M:%S")
    return timeArray.tm_mon   # 2013
# 结果如下
# time.struct_time(tm_year=2013, tm_mon=10, tm_mday=10, tm_hour=23, tm_min=40, tm_sec=0, tm_wday=3, tm_yday=283, tm_isdst=-1)








def get_t1(app_list):
    t1_list=[]

#    print(app_list)
#    app_list=json.dumps(app_list)
#    app_list=json.loads(app_list)
    for app_id in app_list:
        
        t1=get_package_label(app_id,'t1')
        logging.debug(t1)
        if t1.shape[0]<1:
            continue
        t1_list.append(int(t1['t1'].values[0]))
    return set(t1_list)

def get_t2(app_list):
    t2_list=[]

#    logging.debug(app_list)
#    app_list=json.dumps(app_list)
#    app_list=json.loads(app_list)
    for app_id in app_list:
        
        t2=get_package_label(app_id,'t2')
        logging.debug(t2)
        if t2.shape[0]<1:
            continue
        t2_list.append(int(t2['t2'].values[0]))
    return set(t2_list)

def set_t1(app_list):
#    app_list=app_list.split('\'')
    logging.debug(app_list)
    return [x for x in app_list if len(x)>2]

def set_app_dict(app_list):
    app_dict={}
    app_list=json.dumps(app_list)
    app_list=json.loads(app_list)
    logging.debug(app_list)
    
    app_list=app_list.replace('{','').replace('}','').replace('(','').replace(')','').split(', \'')
    for x in app_list:
        x=x.replace('\'','').split(':')
#        logging.debug(x)
#            logging.debug(tmp)
        app_dict[x[0]]=int(x[1])
#            logging.debug(values[0])
    return app_dict
c=0
def compute_date_t1(deviceid_packages,package_label):
    global c
    c=0
#    logging.debug(len(package_label['t1'].unique()))
    columns=[]
    for x in package_label['t1'].unique():
        deviceid_packages['t1_'+str(x)]=0
        columns.append('t1_'+str(x))

    deviceid_packages['add_list']=deviceid_packages['add_list'].apply(lambda x:set_t1(x))
    deviceid_packages['t1_list']=deviceid_packages['add_list'].apply(lambda x:get_t1(x))
    
    for x in package_label['t1'].unique():
        _x=[]
        for i in range(deviceid_packages.shape[0]):
            _x.append({x})
        _x=pd.DataFrame({'a':_x})
        def c(a,b):
#            logging.debug(a,b)
            return len(a&b)>0
        a=list(map(c,_x['a'],deviceid_packages['t1_list']))
#        logging.debug(_x)
#        logging.debug(deviceid_packages['t1_list'])
        filte=np.logical_and(a,True)
#        logging.debug(filte)
        deviceid_packages.ix[filte,'t1_'+str(x)]=1
        
    columns.append('device_id')
    return deviceid_packages.ix[:,columns]
    
    
    
def compute_date_t2(deviceid_packages,package_label):
    global c
    c=0
#    logging.debug(len(package_label['t1'].unique()))
    columns=[]
    for x in package_label['t2'].unique():
        deviceid_packages['t2_'+str(x)]=0
        columns.append('t2_'+str(x))

    deviceid_packages['add_list']=deviceid_packages['add_list'].apply(lambda x:set_t1(x))
    deviceid_packages['t2_list']=deviceid_packages['add_list'].apply(lambda x:get_t2(x))
    
    for x in package_label['t2'].unique():
        _x=[]
        for i in range(deviceid_packages.shape[0]):
            _x.append({x})
        _x=pd.DataFrame({'a':_x})
        def c(a,b):
#            logging.debug(a,b)
            return len(a&b)>0
        a=list(map(c,_x['a'],deviceid_packages['t2_list']))
#        logging.debug(_x)
#        logging.debug(deviceid_packages['t1_list'])
        filte=np.logical_and(a,True)
#        logging.debug(filte)
        deviceid_packages.ix[filte,'t2_'+str(x)]=1
    columns.append('device_id')
    return deviceid_packages.ix[:,columns]
  

def set_app_hout_dict_01(app_list):
    app_dict={}
#    app_list=json.dumps(app_list)
#    app_list=json.loads(app_list)
    app_list=app_list.replace('{','').replace('}','').replace('(','').replace(')','').split('),')
    logging.debug(app_list)
    for x in app_list:
#        logging.debug(x)
        x=x.split(', \'')
#        logging.debug(x)
        for values in x:
            values=values.replace('\'','').split(':')
            tmp=[]
            for _v in values:
                tmp=tmp+_v.split(',') 
#            logging.debug(tmp)
            app_dict[tmp[0]]=int(tmp[1])
#            logging.debug(values[0])
    return app_dict

def set_app_hout_dict_02(app_list):
    app_dict={}
    app_list=app_list.replace('{','').replace('}','').replace('(','').replace(')','').split('),')
    logging.debug(app_list)
    for x in app_list:
#        logging.debug(x)
        x=x.split(', \'')
#        logging.debug(x)
        for values in x:
            values=values.replace('\'','').split(':')
            tmp=[]
            for _v in values:
                tmp=tmp+_v.split(',') 
            logging.debug(tmp)
            if len(tmp)<3:
                continue
            app_dict[tmp[0]]=int(tmp[2])
#            logging.debug(values[0])
    return app_dict

def app_get_hour_t1(app_list):
    t1_dict={}

#    logging.debug(app_list)
#    app_list=json.dumps(app_list)
#    app_list=json.loads(app_list)
    for app_id in app_list.keys():
        
        t2=get_package_label(app_id,'t1')
        logging.debug(t2)
        if t2.shape[0]<1:
            continue
        t1_dict[t2['t1'].values[0]]=app_list[app_id]
    return t1_dict

def compute_date_close_hour(deviceid_packages,package_label):
    global c
    c=0
#    logging.debug(len(package_label['t1'].unique()))
    columns=[]
    for x in package_label['t1'].unique():
        deviceid_packages['close_hour_t1_'+str(x)]=0
        columns.append('close_hour_t1_'+str(x))
#    logging.debug(deviceid_packages.head(2))
    deviceid_packages['app_list']=deviceid_packages['close_hour'].apply(lambda x:set_app_hout_dict_01(x))
    deviceid_packages['t1_dict']=deviceid_packages['app_list'].apply(lambda x:app_get_hour_t1(x))
    
#    logging.debug(deviceid_packages['t1_dict'].head(5))
    for x in package_label['t1'].unique():
        _x=[]
        for i in range(deviceid_packages.shape[0]):
            _x.append(str(x))
        logging.debug(_x)
        _x=pd.DataFrame({'a':_x},dtype='category')

        def c(a,b):
            ert=(str(a) in b.keys())
#            logging.debug(ert)
            return ert
        a=list(map(c,_x['a'],deviceid_packages['t1_dict']))
#        logging.debug(_x)
#        logging.debug(deviceid_packages['t1_list'])
        filte=np.logical_and(a,True)
        def get_values(t1_dict):
            return t1_dict[str(x)]
            
        values=deviceid_packages.ix[filte,'t1_dict'].apply(lambda x:get_values(x))
#        logging.debug(filte)
        deviceid_packages.ix[filte,'close_hour_t1_'+str(x)]=values
        
    #  2
    for x in package_label['t1'].unique():
        deviceid_packages['close_hour_len_t1_'+str(x)]=0
        columns.append('close_hour_len_t1_'+str(x))

    deviceid_packages['app_list']=deviceid_packages['close_hour'].apply(lambda x:set_app_hout_dict_02(x))
    deviceid_packages['t1_dict']=deviceid_packages['app_list'].apply(lambda x:app_get_hour_t1(x))
    
    for x in package_label['t1'].unique():
        _x=[]
        for i in range(deviceid_packages.shape[0]):
            _x.append(str(x))
        _x=pd.DataFrame({'a':_x})
        def c(a,b):
            ert=(str(a) in b.keys())
#            logging.debug(ert)
            return ert
        a=list(map(c,_x['a'],deviceid_packages['t1_dict']))
        logging.debug(_x)
#        logging.debug(deviceid_packages['t1_list'])
        filte=np.logical_and(a,True)

        values=deviceid_packages.ix[filte,'t1_dict'].apply(lambda x:get_values(x))
#        logging.debug(filte)
        deviceid_packages.ix[filte,'close_hour_len_t1_'+str(x)]=values
    columns.append('device_id')
    return deviceid_packages.ix[:,columns]

def compute_date_close_day(deviceid_packages,package_label):
    global c
    c=0
#    logging.debug(len(package_label['t1'].unique()))
    columns=[]
    for x in package_label['t1'].unique():
        deviceid_packages['close_day_t1_'+str(x)]=0
        columns.append('close_day_t1_'+str(x))

    deviceid_packages['app_list']=deviceid_packages['close_day'].apply(lambda x:set_app_dict(x))
    deviceid_packages['t1_dict']=deviceid_packages['app_list'].apply(lambda x:app_get_hour_t1(x))
    
    for x in package_label['t1'].unique():
        _x=[]
        for i in range(deviceid_packages.shape[0]):
            _x.append(str(x))
        _x=pd.DataFrame({'a':_x})
        def c(a,b):
            ert=(str(a) in b.keys())
#            logging.debug(ert)
            return ert
        a=list(map(c,_x['a'],deviceid_packages['t1_dict']))
        logging.debug(_x)
#        logging.debug(deviceid_packages['t1_list'])
        filte=np.logical_and(a,True)
#        logging.debug(filte)
        def get_values(t1_dict):
            return t1_dict[str(x)]
            
        values=deviceid_packages.ix[filte,'t1_dict'].apply(lambda x:get_values(x))
#        logging.debug(filte)
        deviceid_packages.ix[filte,'close_day_t1_'+str(x)]=values
        
    columns.append('device_id')
    return deviceid_packages.ix[:,columns]


def compute_date_close_mon(deviceid_packages,package_label):
    global c
    c=0
    
#    logging.debug(len(package_label['t1'].unique()))
    columns=[]
    for x in package_label['t1'].unique():
        deviceid_packages['close_mon_t1_'+str(x)]=0
        columns.append('close_mon_t1_'+str(x))

    deviceid_packages['app_list']=deviceid_packages['close_mon'].apply(lambda x:set_app_dict(x))
    deviceid_packages['t1_dict']=deviceid_packages['app_list'].apply(lambda x:app_get_hour_t1(x))
    
    for x in package_label['t1'].unique():
        _x=[]
        for i in range(deviceid_packages.shape[0]):
            _x.append(str(x))
        _x=pd.DataFrame({'a':_x})
        def c(a,b):
            ert=(str(a) in b.keys())
#            logging.debug(ert)
            return ert
        a=list(map(c,_x['a'],deviceid_packages['t1_dict']))
        logging.debug(_x)
#        logging.debug(deviceid_packages['t1_list'])
        filte=np.logical_and(a,True)
#        logging.debug(filte)
        def get_values(t1_dict):
            return t1_dict[str(x)]
            
        values=deviceid_packages.ix[filte,'t1_dict'].apply(lambda x:get_values(x))
#        logging.debug(filte)
        deviceid_packages.ix[filte,'close_mon_t1_'+str(x)]=values
        
    columns.append('device_id')
    return deviceid_packages.ix[:,columns]
    
    
def compute_date():
    import multiprocessing

    pool = multiprocessing.Pool(processes=5)
    deviceid_packages_03=pd.read_csv(file_path+'03_deviceid_packages.csv')
    deviceid_packages_04=pd.read_csv(file_path+'04_deviceid_packages.csv')
    package_label=pd.read_csv(file_path+'package_label.csv')
    def app_list(text):
        app_list=text.split('|')
#        print (app_list)
        return app_list
    deviceid_packages=pd.read_csv(file_path+'deviceid_packages.csv')
    deviceid_packages_03['add_list']=deviceid_packages['add_id_list'].apply(lambda line:app_list(line)).tolist()

    
    result = []
    result.append(pool.apply_async(compute_date_t1, (deviceid_packages_03,package_label, )))
    result.append(pool.apply_async(compute_date_t2, (deviceid_packages_03,package_label, )))
    result.append(pool.apply_async(compute_date_close_hour, (deviceid_packages_04,package_label, )))
    result.append(pool.apply_async(compute_date_close_day, (deviceid_packages_04,package_label, )))
    result.append(pool.apply_async(compute_date_close_mon, (deviceid_packages_04,package_label, )))
    pool.close()
    pool.join()
    for res in result:
        deviceid_packages=res.get()
        deviceid_packages_03=pd.merge(deviceid_packages_03,deviceid_packages,on=['device_id'],how='left') 
    
    logging.debug(deviceid_packages_03.head(5))
    
    deviceid_packages_03.to_csv(file_path+'05_deviceid_packages.csv',index= False)
    
    
if __name__=='__main__':
    start_time=time.time()

    compute_date()

# id,
    end_time=time.time()
    logging.debug('耗时:',end_time-start_time)



