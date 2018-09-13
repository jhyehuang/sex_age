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
from db.conn_db import db,cursor,engine,truncate_table,data_from_mysql,get_package_label
from flags import FLAGS, unparsed
from functools import reduce

import time, datetime
import json
import hashlib, csv, math, os, subprocess
import logging
import gc
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.DEBUG)
NR_BINS = 1000000

def hashstr(input):
    return str(int(hashlib.md5(input.encode('utf8')).hexdigest(), 16)%(NR_BINS-1)+1)

file_path=FLAGS.file_path


def time_to_hour(timeStamp):
    # 字符类型的时间
    timeArray = time.localtime(timeStamp)
#    logging.debug(timeArray)
    # 转为时间数组
#    timeArray = time.strptime(timeArray, "%Y-%m-%d %H:%M:%S")
    return str(timeArray.tm_hour)   # 2013

def time_to_week(timeStamp):
    timeArray =int(time.mktime(time.strptime(timeStamp,"%Y%m%d")))
    date = datetime.datetime.fromtimestamp(timeArray)
    day=date.weekday()
#    logging.debug(day)
    return str(day)   # 2013

def time_to_day(timeStamp):
    # 字符类型的时间
    timeArray = time.localtime(timeStamp)
    # 转为时间数组
#    timeArray = time.strptime(timeArray, "%Y-%m-%d %H:%M:%S")
    return str(timeArray.tm_mday)   # 2013

def time_to_mon(timeStamp):
    # 字符类型的时间
    timeArray = time.localtime(timeStamp)
    # 转为时间数组
#    timeArray = time.strptime(timeArray, "%Y-%m-%d %H:%M:%S")
    return str(timeArray.tm_mon)   # 2013
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

def get_max_value_s_key(x):
    if x =={}:
        return 0
    return max(x,key=x.get)

c=0
def devid_hour(deviceid_packages):
    global c
    c=0
#    logging.debug(deviceid_packages.head(5))
    def app_list(text):
        app_list=text.split('|')
#        logging.debug (app_list)
        return app_list
    deviceid_packages['add_list']=deviceid_packages['add_id_list'].apply(lambda line:app_list(line)).tolist()
    def get_max_our(dev_id,app_id_list):
        ret={}
        global c
        c+=1
        for app_id in app_id_list:
            ret[app_id]=get_hour(dev_id,app_id)
#        logging.debug(c,ret)
        return ret
    
    def get_hour(dev_id,app_id_list):
        ret=deviceid_package_start_close_train(dev_id,app_id_list)
        if ret.shape[0]<1:
            return '0','0',0
        ret['close'] = ret['close'].map(int)/1000
        ret['start'] = ret['start'].map(int)/1000

        
        ret['hour'] =ret['close'].apply(lambda x:time_to_hour(x))
        _key_codes = ret['hour'].values
        grp1=ret['close'].groupby(_key_codes)
        cnt1 = grp1.aggregate(np.size)
#        logging.debug(98,cnt1[close.values].values)
        ret['close_size'] = cnt1[_key_codes].values
#        hour=hashstr(''.join(ret['hour'].tolist()))
        hour=max(ret['hour'].tolist())
        time_len=sum(ret['close'].values-ret['start'].values)/60

        ret.fillna(0)
        max_hour=cnt1.max()
        filte=ret['close_size'].values==max_hour
        ret=ret.ix[filte,:]
        max_size_hour=max(ret['hour'].values)

        return hour,max_size_hour,int(time_len)
    logging.debug(deviceid_packages.shape)
    deviceid_packages['close_hour']=deviceid_packages.apply(lambda line:get_max_our(line['device_id'],line['add_list']) ,axis=1)

    logging.debug('============================hour  end==========================')
    logging.debug(deviceid_packages.head(2))
    columns=[]



    deviceid_packages['app_list']=deviceid_packages['close_hour'].apply(lambda x:set_app_dict_01(x))
    deviceid_packages['t1_dict']=deviceid_packages['app_list'].apply(lambda x:app_get_t1(x))
    deviceid_packages['t2_dict']=deviceid_packages['app_list'].apply(lambda x:app_get_t2(x))
    deviceid_packages['close_max_hour_t1']=deviceid_packages['t1_dict'].apply(get_max_value_s_key)
    deviceid_packages['close_max_hour_t2']=deviceid_packages['t2_dict'].apply(get_max_value_s_key)
    
    deviceid_packages['app_list']=deviceid_packages['close_hour'].apply(lambda x:set_app_dict_02(x))
    deviceid_packages['t1_dict']=deviceid_packages['app_list'].apply(lambda x:app_get_t1(x))
    deviceid_packages['t2_dict']=deviceid_packages['app_list'].apply(lambda x:app_get_t2(x))
    deviceid_packages['close_max_size_hour_t1']=deviceid_packages['t1_dict'].apply(get_max_value_s_key)
    deviceid_packages['close_max_size_hour_t2']=deviceid_packages['t2_dict'].apply(get_max_value_s_key)
    
    deviceid_packages['app_list']=deviceid_packages['close_hour'].apply(lambda x:set_app_dict_03(x))
    deviceid_packages['t1_dict']=deviceid_packages['app_list'].apply(lambda x:app_get_t1(x))
    deviceid_packages['close_hour_len_t1']=deviceid_packages['t1_dict'].apply(get_max_value_s_key)
    deviceid_packages['t2_dict']=deviceid_packages['app_list'].apply(lambda x:app_get_t2(x))
    deviceid_packages['close_hour_len_t2']=deviceid_packages['t2_dict'].apply(get_max_value_s_key)
    columns.append('close_max_hour_t1')
    columns.append('close_max_size_hour_t1')
    columns.append('close_hour_len_t1')
    columns.append('close_max_hour_t2')
    columns.append('close_max_size_hour_t2')
    columns.append('close_hour_len_t2')
    columns.append('device_id')
    return deviceid_packages.ix[:,columns]

def devid_day(deviceid_packages):
    global c
    c=0
#    logging.debug(deviceid_packages.head(5))
    def app_list(text):
        app_list=text.split('|')
#        logging.debug (app_list)
        return app_list
    deviceid_packages['add_list']=deviceid_packages['add_id_list'].apply(lambda line:app_list(line)).tolist()
    def get_max_our(dev_id,app_id_list):
        ret={}
        global c
        c+=1
        for app_id in app_id_list:
            ret[app_id]=get_day(dev_id,app_id)
#        logging.debug(c,ret)
        return ret
    
    def get_day(dev_id,app_id_list):
        ret=deviceid_package_start_close_train(dev_id,app_id_list)
#        logging.debug(ret['close'].map(int),ret['start'].map(int))
        if ret.shape[0]<1:
            return '0',0,0
        ret['close'] = ret['close'].map(int)/1000

        
        ret['day'] =ret['close'].apply(lambda x:time_to_day(x))
        _key_codes = ret['day'].values
        grp1=ret['close'].groupby(_key_codes)
        cnt1 = grp1.aggregate(np.size)
#        logging.debug(98,cnt1[close.values].values)
#        ret['close_size'] = cnt1[_key_codes].values
#        logging.debug(ret)
#        ret.fillna(0)
        day=ret['day'].tolist().count('1')
        day_len=len(ret['day'].tolist())
        ret['week_end']=ret['start_day'].apply(lambda x:time_to_week(x))
        week_end=ret['week_end'].tolist()
#        logging.debug(week_end)
        week_end_p=(week_end.count('5')+week_end.count('6'))/len(week_end)
        logging.debug(week_end_p)
#        filte=ret['close_size'].values==max_hour
#        ret=ret.ix[filte,'day'].unique().max()
#        logging.debug(ret)
        
        return day,week_end_p,day_len
    columns=[]
    deviceid_packages['close_day']=deviceid_packages.apply(lambda line:get_max_our(line['device_id'],line['add_list']) ,axis=1)
#    deviceid_packages.to_csv(file_path+'0402_deviceid_train.csv')
    logging.debug('============================day  end==========================')
    deviceid_packages['app_list']=deviceid_packages['close_day'].apply(lambda x:set_app_dict_01(x))
    deviceid_packages['t1_dict']=deviceid_packages['app_list'].apply(lambda x:app_get_t1(x))
    deviceid_packages['close_max_day_t1']=deviceid_packages['t1_dict'].apply(get_max_value_s_key)
    deviceid_packages['t2_dict']=deviceid_packages['app_list'].apply(lambda x:app_get_t2(x))
    deviceid_packages['close_max_day_t2']=deviceid_packages['t2_dict'].apply(get_max_value_s_key)
    
    deviceid_packages['app_list']=deviceid_packages['close_day'].apply(lambda x:set_app_dict_02(x))
    deviceid_packages['t1_dict']=deviceid_packages['app_list'].apply(lambda x:app_get_t1(x))
    deviceid_packages['close_max_size_day_t1']=deviceid_packages['t1_dict'].apply(get_max_value_s_key)
    deviceid_packages['t2_dict']=deviceid_packages['app_list'].apply(lambda x:app_get_t2(x))
    deviceid_packages['close_max_size_day_t2']=deviceid_packages['t2_dict'].apply(get_max_value_s_key)
     
    deviceid_packages['app_list']=deviceid_packages['close_day'].apply(lambda x:set_app_dict_03(x))
    deviceid_packages['t1_dict']=deviceid_packages['app_list'].apply(lambda x:app_get_t1(x))
    deviceid_packages['close_day_len_t1']=deviceid_packages['t1_dict'].apply(get_max_value_s_key)
    deviceid_packages['t2_dict']=deviceid_packages['app_list'].apply(lambda x:app_get_t2(x))
    deviceid_packages['close_day_len_t2']=deviceid_packages['t2_dict'].apply(get_max_value_s_key)
    columns.append('close_max_day_t1')
    columns.append('close_max_size_day_t1')
    columns.append('close_day_len_t1')
    columns.append('close_max_day_t2')
    columns.append('close_max_size_day_t2')
    columns.append('close_day_len_t2')
    columns.append('device_id')
    return deviceid_packages.ix[:,columns]
    
def devid_mon(deviceid_packages):
    global c
    c=0

#    logging.debug(deviceid_packages.head(5))
    def app_list(text):
        app_list=text.split('|')
#        logging.debug (app_list)
        return app_list
    deviceid_packages['add_list']=deviceid_packages['add_id_list'].apply(lambda line:app_list(line)).tolist()
    def get_max_our(dev_id,app_id_list):
        ret={}
        global c
        c+=1
        for app_id in app_id_list:
            ret[app_id]=get_mon(dev_id,app_id)
#        logging.debug(c,ret)
        return ret
    
    def get_mon(dev_id,app_id_list):
        ret=deviceid_package_start_close_train(dev_id,app_id_list)
#        logging.debug(ret['close'].map(int),ret['start'].map(int))
        if ret.shape[0]<1:
            return '0',0
        ret['close'] = ret['close'].map(int)/1000

        
        ret['mon'] =ret['close'].apply(lambda x:time_to_mon(x))
        _key_codes = ret['mon'].values
        grp1=ret['close'].groupby(_key_codes)
        cnt1 = grp1.aggregate(np.size)
#        logging.debug(98,cnt1[close.values].values)
#        ret['close_size'] = cnt1[_key_codes].values
#        logging.debug(ret)
#        ret.fillna(0)
        mon=hashstr(''.join(ret['mon'].tolist()))
        mon_len=len(ret['mon'].tolist())
#        filte=ret['close_size'].values==max_hour
#        ret=ret.ix[filte,'mon'].unique().max()
#        logging.debug(ret)
        return mon,mon_len
    columns=[]
    deviceid_packages['close_mon']=deviceid_packages.apply(lambda line:get_max_our(line['device_id'],line['add_list']) ,axis=1)
    
#    deviceid_packages.to_csv(file_path+'0403_deviceid_train.csv')
    print('============================mon  end==========================')
    deviceid_packages['app_list']=deviceid_packages['close_mon'].apply(lambda x:set_app_dict_02(x))
    deviceid_packages['t1_dict']=deviceid_packages['app_list'].apply(lambda x:app_get_t1(x))
    deviceid_packages['close_mon_len_t1']=deviceid_packages['t1_dict'].apply(get_max_value_s_key)
    deviceid_packages['t2_dict']=deviceid_packages['app_list'].apply(lambda x:app_get_t2(x))
    deviceid_packages['close_mon_len_t2']=deviceid_packages['t2_dict'].apply(get_max_value_s_key)
    columns.append('close_mon_len_t1')
    columns.append('close_mon_len_t2')
    columns.append('device_id')
    return deviceid_packages.ix[:,columns]
    
def set_app_dict_01(app_list):
    app_dict={}
    logging.debug(app_list)
    for x,x_value in app_list.items():
        app_dict[x]=int(x_value[0])
    return app_dict

def set_app_dict_02(app_list):
    app_dict={}
    logging.debug(app_list)
    for x,x_value in app_list.items():
        app_dict[x]=int(x_value[1])
    return app_dict

def set_app_dict_03(app_list):
    app_dict={}
    logging.debug(app_list)
    for x,x_value in app_list.items():
        app_dict[x]=int(x_value[2])
    return app_dict

def app_get_t1(app_list):
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

def app_get_t2(app_list):
    t1_dict={}

#    logging.debug(app_list)
#    app_list=json.dumps(app_list)
#    app_list=json.loads(app_list)
    for app_id in app_list.keys():
        
        t2=get_package_label(app_id,'t2')
        logging.debug(t2)
        if t2.shape[0]<1:
            continue
        t1_dict[t2['t2'].values[0]]=app_list[app_id]
    return t1_dict





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

    logging.debug('============================================')
    deviceid_packages=pd.merge(result[0].get(),result[1].get(),on=['device_id'],how='left') 
    deviceid_packages=pd.merge(deviceid_packages,result[2].get(),on=['device_id'],how='left') 
    
    logging.debug(deviceid_packages.head(5))
    
#    deviceid_packages.to_csv(file_path+'04_deviceid_packages.csv', columns=['device_id','close_hour','close_day','close_mon'],index= False)
#    deviceid_packages.to_csv(file_path+'08_deviceid_packages.csv',columns=columns,index= False)
    deviceid_packages.to_csv(file_path+'04_deviceid_packages.csv',index= False)
    
if __name__=='__main__':
    start_time=time.time()

    compute_date()

# id,
    end_time=time.time()
    logging.debug('耗时:'+str(end_time-start_time))



