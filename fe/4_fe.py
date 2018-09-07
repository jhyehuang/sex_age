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

import time, datetime
import json


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



def deviceid_package_start_close_train(deviceid,app_id='0'):
    sql='select * from deviceid_package_start_close where device_id=\"'
    sql=sql+deviceid+'\" '
    sql=sql+'and app_id=\"'
    sql=sql+app_id+'\"'
    ret=data_from_mysql(sql)
    return ret


def dev_id_train():
    sql='select * from deviceid_train'
    deviceid_train=data_from_mysql(sql)
    return deviceid_train

c=0
def devid_hour(deviceid_packages):
    global c
    c=0
#    print(deviceid_packages.head(5))
    def app_list(text):
        app_list=text.split('|')
#        print (app_list)
        return app_list
    deviceid_packages['add_list']=deviceid_packages['add_id_list'].apply(lambda line:app_list(line)).tolist()
    def get_max_our(dev_id,app_id_list):
        ret={}
        global c
        c+=1
        for app_id in app_id_list:
            ret[app_id]=get_hour(dev_id,app_id)
        print(c,ret)
        return ret
    
    def get_hour(dev_id,app_id_list):
        ret=deviceid_package_start_close_train(dev_id,app_id_list)
        if ret.shape[0]<1:
            return 0
        ret['close'] = ret['close'].map(int)/1000
        ret['start'] = ret['start'].map(int)/1000

        
        ret['hour'] =ret['close'].apply(lambda x:time_to_hour(x))
        _key_codes = ret['hour'].values
        grp1=ret['close'].groupby(_key_codes)
        cnt1 = grp1.aggregate(np.size)
#        print(98,cnt1[close.values].values)
        ret['close_size'] = cnt1[_key_codes].values
#        print(ret)
        ret.fillna(0)
        max_hour=cnt1.max()
        filte=ret['close_size'].values==max_hour
        ret=ret.ix[filte,:]
        hour=ret['hour'].unique().max()
        time_len=min(sum(ret['close'].values-ret['start'].values)/60,400)
#        print(time_len)
        
        return hour,int(time_len)
    print(deviceid_packages.shape)
    deviceid_packages['close_hour']=deviceid_packages.apply(lambda line:get_max_our(line['device_id'],line['add_list']) ,axis=1)
#    deviceid_packages.to_csv(file_path+'0401_deviceid_train.csv')
    print('============================hour  end==========================')
    print(deviceid_packages.head(2))
    columns=['device_id','close_hour',]
    return deviceid_packages


def devid_day(deviceid_packages):
    global c
    c=0
#    print(deviceid_packages.head(5))
    def app_list(text):
        app_list=text.split('|')
#        print (app_list)
        return app_list
    deviceid_packages['add_list']=deviceid_packages['add_id_list'].apply(lambda line:app_list(line)).tolist()
    def get_max_our(dev_id,app_id_list):
        ret={}
        global c
        c+=1
        for app_id in app_id_list:
            ret[app_id]=get_day(dev_id,app_id)
        print(c,ret)
        return ret
    
    def get_day(dev_id,app_id_list):
        ret=deviceid_package_start_close_train(dev_id,app_id_list)
#        print(ret['close'].map(int),ret['start'].map(int))
        if ret.shape[0]<1:
            return 0
        ret['close'] = ret['close'].map(int)/1000

        
        ret['day'] =ret['close'].apply(lambda x:time_to_day(x))
        _key_codes = ret['day'].values
        grp1=ret['close'].groupby(_key_codes)
        cnt1 = grp1.aggregate(np.size)
#        print(98,cnt1[close.values].values)
#        ret['close_size'] = cnt1[_key_codes].values
#        print(ret)
#        ret.fillna(0)
        max_hour=cnt1.max()
#        filte=ret['close_size'].values==max_hour
#        ret=ret.ix[filte,'day'].unique().max()
#        print(ret)
        
        return int(max_hour)
    
    deviceid_packages['close_day']=deviceid_packages.apply(lambda line:get_max_our(line['device_id'],line['add_list']) ,axis=1)
#    deviceid_packages.to_csv(file_path+'0402_deviceid_train.csv')
    print('============================day  end==========================')
    print(deviceid_packages.head(2))
    columns=['device_id','close_day',]
    return deviceid_packages
    
def devid_mon(deviceid_packages):
    global c
    c=0

#    print(deviceid_packages.head(5))
    def app_list(text):
        app_list=text.split('|')
#        print (app_list)
        return app_list
    deviceid_packages['add_list']=deviceid_packages['add_id_list'].apply(lambda line:app_list(line)).tolist()
    def get_max_our(dev_id,app_id_list):
        ret={}
        global c
        c+=1
        for app_id in app_id_list:
            ret[app_id]=get_mon(dev_id,app_id)
        print(c,ret)
        return ret
    
    def get_mon(dev_id,app_id_list):
        ret=deviceid_package_start_close_train(dev_id,app_id_list)
#        print(ret['close'].map(int),ret['start'].map(int))
        if ret.shape[0]<1:
            return 0
        ret['close'] = ret['close'].map(int)/1000

        
        ret['mon'] =ret['close'].apply(lambda x:time_to_mon(x))
        _key_codes = ret['mon'].values
        grp1=ret['close'].groupby(_key_codes)
        cnt1 = grp1.aggregate(np.size)
#        print(98,cnt1[close.values].values)
#        ret['close_size'] = cnt1[_key_codes].values
#        print(ret)
#        ret.fillna(0)
        max_hour=cnt1.max()
#        filte=ret['close_size'].values==max_hour
#        ret=ret.ix[filte,'mon'].unique().max()
#        print(ret)
        return int(max_hour)
    
    deviceid_packages['close_mon']=deviceid_packages.apply(lambda line:get_max_our(line['device_id'],line['add_list']) ,axis=1)
    
#    deviceid_packages.to_csv(file_path+'0403_deviceid_train.csv')
    print('============================mon  end==========================')
    print(deviceid_packages.head(2))
    columns=['device_id','close_mon',]
    return deviceid_packages
    
def compute_date():
    import multiprocessing

    pool = multiprocessing.Pool(processes=3)
    deviceid_packages=pd.read_csv(file_path+'deviceid_packages.csv')
#    deviceid_train=dev_id_train()
    
    result = []
    result.append(pool.apply_async(devid_hour, (deviceid_packages, )))
    result.append(pool.apply_async(devid_day, (deviceid_packages, )))
    result.append(pool.apply_async(devid_mon, (deviceid_packages, )))
    pool.close()
    pool.join()
    for res in result:
        ret=res.get()
        print('============================================',ret.head(2))
        deviceid_packages=pd.merge(deviceid_packages,ret,on=['device_id'],how='left') 
    
    print(deviceid_packages.head(5))
    
    deviceid_packages.to_csv(file_path+'04_deviceid_packages.csv', columns=['device_id','close_hour','close_day','close_mon'],index= False)
    
    
if __name__=='__main__':
    start_time=time.time()

    compute_date()

# id,
    end_time=time.time()
    print('耗时:',end_time-start_time)



