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
from db.conn_db import db,cursor,engine,truncate_table,data_from_mysql,get_package_dict
from flags import FLAGS, unparsed
from functools import reduce

import time, datetime
import json

import logging

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.DEBUG)



file_path=FLAGS.file_path

def def_time_label(hour):
    if hour>=18 and hour<=24:
        return 4
    elif hour>=0 and hour<8:
        return 0
    elif hour>=8 and hour<12:
        return 1
    elif hour>=12 and hour<14:
        return 2
    elif hour>=14 and hour<18:
        return 3

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

def get_dev_hour_info_t1(dev_id,app_id_list):
    logging.debug(app_id_list)
    if len(app_id_list)<1:
        return {}
    tx_list=[]
    for app_id in app_id_list:
        t1_dict={}
        
        t2=get_package_dict(app_id,'t1,t2')
        
        if len(t2)<1:
#            print(t2)
            continue
        app_hour_info=get_hour_info(dev_id,app_id)
        if len(app_hour_info)<1:
#            print(t2)
            continue
        t1_dict['t1']=t2.get('t1','0')
        t1_dict['t2']=t2.get('t2','0')
        t1_dict['app_id']=app_id
        t1_dict.update(app_hour_info)
        tx_list.append(t1_dict)
    tx_pd=pd.DataFrame(x for x in tx_list)
    if tx_pd.shape[0]<1:
        return {}

    result_t1={}
    for t1 in tx_pd.t1.unique():
        tmp_dict={}
        tmp_dict['start_hour_len']=tx_pd.ix[tx_pd.t1.values==t1,'start_hour_len'].sum()
        tmp_dict['start_hour_size']=tx_pd.ix[tx_pd.t1.values==t1,'start_hour_size'].sum()
        tmp_dict['close_hour']=''.join(tx_pd.ix[tx_pd.t1.values==t1,'close_hour'].tolist())
        tmp_dict['start_hour']=''.join(tx_pd.ix[tx_pd.t1.values==t1,'start_hour'].tolist())
        result_t1[t1]=tmp_dict
    logging.debug(result_t1)
    return result_t1

def get_dev_hour_info_t2(dev_id,app_id_list):
    logging.debug(app_id_list)
    if len(app_id_list)<1:
        return {}
    tx_list=[]
    for app_id in app_id_list:
        t2_dict={}
        
        t2=get_package_dict(app_id,'t1,t2')
        
        if len(t2)<1:
#            print(t2)
            continue
        app_hour_info=get_hour_info(dev_id,app_id)
        if len(app_hour_info)<1:
#            print(t2)
            continue
        t2_dict['t1']=t2.get('t1','0')
        t2_dict['t2']=t2.get('t2','0')
        t2_dict['app_id']=app_id
        t2_dict.update(app_hour_info)
        tx_list.append(t2_dict)
    tx_pd=pd.DataFrame(x for x in tx_list)
    if tx_pd.shape[0]<1:
        return {}

    result_t2={}
    for t2 in tx_pd.t2.unique():
        tmp_dict={}
        tmp_dict['start_hour_len']=tx_pd.ix[tx_pd.t2.values==t2,'start_hour_len'].sum()
        tmp_dict['start_hour_size']=tx_pd.ix[tx_pd.t2.values==t2,'start_hour_size'].sum()
        tmp_dict['close_hour']=''.join(tx_pd.ix[tx_pd.t2.values==t2,'close_hour'].tolist())
        tmp_dict['start_hour']=''.join(tx_pd.ix[tx_pd.t2.values==t2,'start_hour'].tolist())
        result_t2[t2]=tmp_dict
    logging.debug(result_t2)
    return result_t2

def get_hour_info(dev_id,app_id_list):
    result={}
    ret=deviceid_package_start_close_train(dev_id,app_id_list)
    if ret.shape[0]<1:
        return 0
    result['start_hour_len']=sum(ret['close'].map(int)/1000-ret['start'].map(int)/1000)
    ret['close'] = ret['close'].map(int)/1000
    ret['start'] = ret['start'].map(int)/1000

    
    result['close_hour'] =''.join(ret['close'].apply(lambda x:time_to_hour(x)).tolist())
    result['start_hour'] =''.join(ret['start'].apply(lambda x:time_to_hour(x)).tolist())
    result['start_hour_size']=ret.shape[0]
    return result

c=0
def devid_hour(deviceid_packages,package_label):
    global c
    c=0
#    print(deviceid_packages.head(5))
    def app_list(text):
        app_list=text.split('|')
#        print (app_list)
        return app_list
    deviceid_packages['add_list']=deviceid_packages['add_id_list'].apply(lambda line:app_list(line)).tolist()
    deviceid_packages['t1_hour_time']=deviceid_packages.apply(lambda line:get_dev_hour_info_t1(line['device_id'],line['add_list']),axis=1)
    deviceid_packages['t2_hour_time']=deviceid_packages.apply(lambda line:get_dev_hour_info_t2(line['device_id'],line['add_list']),axis=1)
    
    columns=[]
    logging.debug(FLAGS.t1_feature.replace('\'','').split(','))
    for x in FLAGS.t1_feature.replace('\'','').split(','):
        for suffix in ['close_hour','start_hour','start_hour_len','start_hour_size']:
            columns.append('hour_t1_'+suffix+'_'+str(x))
    for x in FLAGS.t2_feature.replace('\'','').split(','):
        
            columns.append('hour_t2_'+suffix+'_'+str(x))
        
    for x in package_label['t1'].unique():
        for suffix in ['close_hour','start_hour','start_hour_len','start_hour_size']:
            deviceid_packages['hour_t1_'+suffix+'_'+str(x)]=int(0)

    for x in package_label['t2'].unique():
        for suffix in ['close_hour','start_hour','start_hour_len','start_hour_size']:
            deviceid_packages['hour_t2_'+suffix+'_'+str(x)]=int(0)

    
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
        a=list(map(c,_x['a'],deviceid_packages['t1_hour_time']))


        filte=np.logical_and(a,True)
        def get_values(t1_dict):
            return t1_dict[str(x)]
        def get_sub_values(v_dict,col):
            return v_dict[col]
            
        values=deviceid_packages.ix[filte,'t1_hour_time'].apply(lambda x:get_values(x))
#        print(filte)
        deviceid_packages.ix[filte,'hour_t1_close_hour_'+str(x)]=values.apply(lambda x:get_sub_values(x,'close_hour'))
        deviceid_packages.ix[filte,'hour_t1_start_hour_'+str(x)]=values.apply(lambda x:get_sub_values(x,'start_hour'))
        deviceid_packages.ix[filte,'hour_t1_start_hour_len_'+str(x)]=values.apply(lambda x:get_sub_values(x,'start_hour_len'))
        deviceid_packages.ix[filte,'hour_t1_start_hour_size_'+str(x)]=values.apply(lambda x:get_sub_values(x,'start_hour_size'))

    for x in package_label['t2'].unique():
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
        a=list(map(c,_x['a'],deviceid_packages['t2_hour_time']))


        filte=np.logical_and(a,True)
            
        values=deviceid_packages.ix[filte,'t2_hour_time'].apply(lambda x:get_values(x))
#        print(filte)
        deviceid_packages.ix[filte,'hour_t2_close_hour_'+str(x)]=values.apply(lambda x:get_sub_values(x,'close_hour'))
        deviceid_packages.ix[filte,'hour_t2_start_hour_'+str(x)]=values.apply(lambda x:get_sub_values(x,'start_hour'))
        deviceid_packages.ix[filte,'hour_t2_start_hour_len_'+str(x)]=values.apply(lambda x:get_sub_values(x,'start_hour_len'))
        deviceid_packages.ix[filte,'hour_t2_start_hour_size_'+str(x)]=values.apply(lambda x:get_sub_values(x,'start_hour_size'))

    columns.append('device_id')
    logging.debug(columns)
    
    return deviceid_packages.ix[:, columns]

def get_dev_day_info_t1(dev_id,app_id_list):
    logging.debug(app_id_list)
    if len(app_id_list)<1:
        return {}
    tx_list=[]
    for app_id in app_id_list:
        t1_dict={}
        
        t2=get_package_dict(app_id,'t1,t2')
        
        if len(t2)<1:
#            print(t2)
            continue
        app_day_info=get_day_info(dev_id,app_id)
        if len(app_day_info)<1:
#            print(t2)
            continue
        t1_dict['t1']=t2.get('t1','0')
        t1_dict['t2']=t2.get('t2','0')
        t1_dict['app_id']=app_id
        t1_dict.update(app_day_info)
        tx_list.append(t1_dict)
    tx_pd=pd.DataFrame(x for x in tx_list)
    if tx_pd.shape[0]<1:
        return {}

    result_t1={}
    for t1 in tx_pd.t1.unique():
        tmp_dict={}
        tmp_dict['start_day_len']=tx_pd.ix[tx_pd.t1.values==t1,'start_day_len'].sum()
        tmp_dict['start_day_size']=tx_pd.ix[tx_pd.t1.values==t1,'start_day_size'].sum()
        tmp_dict['close_day']=''.join(tx_pd.ix[tx_pd.t1.values==t1,'close_day'].tolist())
        tmp_dict['start_day']=''.join(tx_pd.ix[tx_pd.t1.values==t1,'start_day'].tolist())
        result_t1[t1]=tmp_dict
    logging.debug(result_t1)
    return result_t1

def get_dev_day_info_t2(dev_id,app_id_list):
    logging.debug(app_id_list)
    if len(app_id_list)<1:
        return {}
    tx_list=[]
    for app_id in app_id_list:
        t2_dict={}
        
        t2=get_package_dict(app_id,'t1,t2')
        
        if len(t2)<1:
#            print(t2)
            continue
        app_day_info=get_day_info(dev_id,app_id)
        if len(app_day_info)<1:
#            print(t2)
            continue
        t2_dict['t1']=t2.get('t1','0')
        t2_dict['t2']=t2.get('t2','0')
        t2_dict['app_id']=app_id
        t2_dict.update(app_day_info)
        tx_list.append(t2_dict)
    tx_pd=pd.DataFrame(x for x in tx_list)
    if tx_pd.shape[0]<1:
        return {}

    result_t2={}
    for t2 in tx_pd.t2.unique():
        tmp_dict={}
        tmp_dict['start_day_len']=tx_pd.ix[tx_pd.t2.values==t2,'start_day_len'].sum()
        tmp_dict['start_day_size']=tx_pd.ix[tx_pd.t2.values==t2,'start_day_size'].sum()
        tmp_dict['close_day']=''.join(tx_pd.ix[tx_pd.t2.values==t2,'close_day'].tolist())
        tmp_dict['start_day']=''.join(tx_pd.ix[tx_pd.t2.values==t2,'start_day'].tolist())
        result_t2[t2]=tmp_dict
    logging.debug(result_t2)
    return result_t2

def get_day_info(dev_id,app_id_list):
    result={}
    ret=deviceid_package_start_close_train(dev_id,app_id_list)
    if ret.shape[0]<1:
        return 0
    result['start_day_len']=sum(ret['close'].map(int)/1000-ret['start'].map(int)/1000)/60
    
    result['close_day'] =''.join(ret['close'].apply(lambda x:time_to_day(x)).tolist())
    result['start_day'] =''.join(ret['start'].apply(lambda x:time_to_day(x)).tolist())
    result['start_day_size']=ret.shape[0]
    return result


def devid_day(deviceid_packages,package_label):
    global c
    c=0
#    print(deviceid_packages.head(5))
    def app_list(text):
        app_list=text.split('|')
#        print (app_list)
        return app_list
    deviceid_packages['add_list']=deviceid_packages['add_id_list'].apply(lambda line:app_list(line)).tolist()
    deviceid_packages['t1_day_time']=deviceid_packages.apply(lambda line:get_dev_day_info_t1(line['device_id'],line['add_list']),axis=1)
    deviceid_packages['t2_day_time']=deviceid_packages.apply(lambda line:get_dev_day_info_t2(line['device_id'],line['add_list']),axis=1)
    
    columns=[]
    logging.debug(FLAGS.t1_feature.replace('\'','').split(','))
    for x in FLAGS.t1_feature.replace('\'','').split(','):
        for suffix in ['close_day','start_day','start_day_len','start_day_size']:
            columns.append('day_t1_'+suffix+'_'+str(x))
    for x in FLAGS.t2_feature.replace('\'','').split(','):
        for suffix in ['close_day','start_day','start_day_len','start_day_size']:
            columns.append('day_t2_'+suffix+'_'+str(x))
        
    for x in package_label['t1'].unique():
        for suffix in ['close_day','start_day','start_day_len','start_day_size']:
            deviceid_packages['day_t1_'+suffix+'_'+str(x)]=int(0)

    for x in package_label['t2'].unique():
        for suffix in ['close_day','start_day','start_day_len','start_day_size']:
            deviceid_packages['day_t2_'+suffix+'_'+str(x)]=int(0)

    
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
        a=list(map(c,_x['a'],deviceid_packages['t1_day_time']))


        filte=np.logical_and(a,True)
        def get_values(t1_dict):
            return t1_dict[str(x)]
        def get_sub_values(v_dict,col):
            return v_dict[col]
            
        values=deviceid_packages.ix[filte,'t1_day_time'].apply(lambda x:get_values(x))
#        print(filte)
        deviceid_packages.ix[filte,'day_t1_close_hour_'+str(x)]=values.apply(lambda x:get_sub_values(x,'close_day'))
        deviceid_packages.ix[filte,'day_t1_start_hour_'+str(x)]=values.apply(lambda x:get_sub_values(x,'start_day'))
        deviceid_packages.ix[filte,'day_t1_start_hour_len_'+str(x)]=values.apply(lambda x:get_sub_values(x,'start_day_len'))
        deviceid_packages.ix[filte,'day_t1_start_hour_size_'+str(x)]=values.apply(lambda x:get_sub_values(x,'start_day_size'))

    for x in package_label['t2'].unique():
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
        a=list(map(c,_x['a'],deviceid_packages['t2_day_time']))


        filte=np.logical_and(a,True)
            
        values=deviceid_packages.ix[filte,'t2_day_time'].apply(lambda x:get_values(x))
#        print(filte)
        deviceid_packages.ix[filte,'hour_t2_close_hour_'+str(x)]=values.apply(lambda x:get_sub_values(x,'close_hour'))
        deviceid_packages.ix[filte,'hour_t2_start_hour_'+str(x)]=values.apply(lambda x:get_sub_values(x,'start_hour'))
        deviceid_packages.ix[filte,'hour_t2_start_hour_len_'+str(x)]=values.apply(lambda x:get_sub_values(x,'start_hour_len'))
        deviceid_packages.ix[filte,'hour_t2_start_hour_size_'+str(x)]=values.apply(lambda x:get_sub_values(x,'start_hour_size'))

    columns.append('device_id')
    logging.debug(columns)
    
    return deviceid_packages.ix[:, columns]
    
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
        max_hour=cnt1.mean()
#        filte=ret['close_size'].values==max_hour
#        ret=ret.ix[filte,'mon'].unique().max()
#        print(ret)
        return int(sum(cnt1)/max_hour)
    
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



