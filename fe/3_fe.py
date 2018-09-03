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
from db.conn_db import db,cursor,engine,truncate_table,data_from_mysql
from flags import FLAGS, unparsed
from functools import reduce

file_path=FLAGS.file_path

def deviceid_package_start_close_train(deviceid,app_id='0'):
    sql='select * from deviceid_package_start_close where device_id=\"'
    sql=sql+deviceid+'\" '
    sql=sql+'and app_id=\"'
    sql=sql+app_id+'\"'
#    print(sql)
    ret=data_from_mysql(sql)
    return ret


def dev_id_train():
    sql='select * from deviceid_train'
    deviceid_train=data_from_mysql(sql)
    return deviceid_train

c=0
def devid_times():
    
    global c
    c=0
    deviceid_packages=pd.read_csv(file_path+'deviceid_packages.csv')
    deviceid_train=dev_id_train()
#    print(deviceid_packages.head(5))
    def app_list(text):
        app_list=text.split('|')
#        print (app_list)
        return app_list
    deviceid_packages['add_list']=deviceid_packages['add_id_list'].apply(lambda line:app_list(line)).tolist()
    def get_times_len(dev_id,app_id_list):
        ret=0
        global c
        c+=1
        for app_id in app_id_list:
            ret=ret+get_times(dev_id,app_id)
        print(c,dev_id,ret)
        return min(int(ret),400)
    
    def get_times(dev_id,app_id_list):
        ret=deviceid_package_start_close_train(dev_id,app_id_list)
#        print(ret['close'].map(int),ret['start'].map(int))
        if ret.shape[0]<1:
            return 0
        ret['close'] = ret['close'].map(int)
        ret['start'] = ret['start'].map(int)
#        print(ret)
        condition = sum(ret['close'].map(int).values-ret['start'].map(int).values)/1000/60/60
        return min(int(condition),400)
#    print(deviceid_packages.shape)
     
    deviceid_packages['times_len']=deviceid_packages.apply(lambda line:get_times_len(line['device_id'],line['add_list']) ,axis=1)


    
    

    
#    deviceid_train=pd.merge(deviceid_train,deviceid_packages,on=['device_id'],how='left') 
    
#    print(deviceid_packages.head(5))
    
    deviceid_packages.to_csv(file_path+'03_deviceid_packages.csv', columns=['device_id','add_list','times_len'],index= False)
    
    
if __name__=='__main__':
    start_time=time.time()

    devid_times()

# id,
    end_time=time.time()
    print('耗时:',end_time-start_time)



