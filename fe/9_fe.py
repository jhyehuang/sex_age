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
from db.conn_db import db,cursor,engine,truncate_table,data_from_mysql,dev_id_train
from flags import FLAGS, unparsed
from functools import reduce
import logging
import operator  
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.DEBUG)



file_path=FLAGS.file_path


def intersection_list(app_list):
    import copy
    retlist=[]
    for i in range(0,len(app_list)):
        
        alist=app_list[i]
        tlist=copy.deepcopy(app_list)
        tlist.remove(alist)
        blist=reduce(operator.add, tlist)
        clist=list(set(alist).intersection(set(blist)))
        retlist.append(clist)
    return retlist

def app_dict_get(app_list,app_dict):
    for x in app_list:
        if x in app_dict:
            return app_dict.get(x,0)
        
def app3_w(deviceid_packages,deviceid_train):
#    deviceid_train=deviceid_train.fillna(-1)
    apps_list=[]
    all_app=reduce(operator.add, deviceid_packages.app_list.tolist())
    train_app=reduce(operator.add, deviceid_train.app_list.tolist())
    no_train_app=list(set(all_app).difference(set(train_app)))
    for i in range(0,23):
        type_list1=deviceid_train.ix[deviceid_train['n_class'].values==i,'app_list'].tolist()
#        print(i)
        if len(type_list1)<1:
            apps_list.append([])
        else:
            type_list1=reduce(operator.add, type_list1)
            apps_list.append(type_list1)
    
    inter_list=intersection_list(apps_list)
    app_dict={}
    def c(a,b):
#        print(a,b)
        ert=(a in b)
        return ert
    for i,x in enumerate(inter_list):
        for li in x:
            _x=[]
            for _ in range(deviceid_train.shape[0]):
                _x.append(li)
    #        print(_x)
            _x=pd.DataFrame({'a':_x},dtype='category')
            a=list(map(c,_x['a'],deviceid_train['app_list']))
            filte1=np.logical_and(deviceid_train.n_class==i,a)
            print(np.unique(filte1))
            filte2=np.logical_and(a,True)
            if li in app_dict:
                app_dict[li]=max(app_dict[li],deviceid_train.ix[filte1,'app_list'].shape[0]/deviceid_train.ix[filte2,'app_list'].shape[0])
            else:
                app_dict[li]=deviceid_train.ix[filte1,'app_list'].shape[0]/deviceid_train.ix[filte2,'app_list'].shape[0]
    deviceid_packages['app3_w']=deviceid_packages['app_list'].apply(lambda x:app_dict_get(x,app_dict))
    return  deviceid_packages.ix[:,['device_id','app3_w']]

        
def app1_w(deviceid_packages,deviceid_train):
#    deviceid_train=deviceid_train.fillna(-1)
    apps_list=[]
    all_app=reduce(operator.add, deviceid_packages.app_list.tolist())
#    print(deviceid_train.app_list.tolist())
    train_app=reduce(operator.add, deviceid_train.app_list.tolist())
    no_train_app=list(set(all_app).difference(set(train_app)))
    for i in [1,2]:
        type_list1=deviceid_train.ix[deviceid_train['sex'].values==i,'app_list'].tolist()
        print(i)
        if len(type_list1)<1:
            apps_list.append([])
        else:
            type_list1=reduce(operator.add, type_list1)
            apps_list.append(type_list1)
    
    inter_list=intersection_list(apps_list)
    app_dict={}
    def c(a,b):
#        print(a,b)
        ert=(a in b)
        return ert
    for i,x in enumerate(inter_list):
        for li in x:
            _x=[]
            for _ in range(deviceid_train.shape[0]):
                _x.append(li)
    #        print(_x)
            _x=pd.DataFrame({'a':_x},dtype='category')
            a=list(map(c,_x['a'],deviceid_train['app_list']))
            filte1=np.logical_and(deviceid_train.n_class==i,a)
            print(np.unique(filte1))
            filte2=np.logical_and(a,True)
            if li in app_dict:
                app_dict[li]=min(app_dict[li],deviceid_train.ix[filte1,'app_list'].shape[0]/deviceid_train.ix[filte2,'app_list'].shape[0])
            else:
                app_dict[li]=deviceid_train.ix[filte1,'app_list'].shape[0]/deviceid_train.ix[filte2,'app_list'].shape[0]
    deviceid_packages['app1_w']=deviceid_packages['app_list'].apply(lambda x:app_dict_get(x,app_dict))
    return  deviceid_packages.ix[:,['device_id','app1_w']]
    
def app2_w(deviceid_packages,deviceid_train):
#    deviceid_train=deviceid_train.fillna(-1)
    apps_list=[]
    all_app=reduce(operator.add, deviceid_packages.app_list.tolist())
    train_app=reduce(operator.add, deviceid_train.app_list.tolist())
    no_train_app=list(set(all_app).difference(set(train_app)))
    for i in range(0,11):
        type_list1=deviceid_train.ix[deviceid_train['age'].values==i,'app_list'].tolist()
        print(i)
        if len(type_list1)<1:
            apps_list.append([])
        else:
            type_list1=reduce(operator.add, type_list1)
            apps_list.append(type_list1)
    
    inter_list=intersection_list(apps_list)
    app_dict={}
    def c(a,b):
#        print(a,b)
        ert=(a in b)
        return ert
    for i,x in enumerate(inter_list):
        for li in x:
            _x=[]
            for _ in range(deviceid_train.shape[0]):
                _x.append(li)
    #        print(_x)
            _x=pd.DataFrame({'a':_x},dtype='category')
            a=list(map(c,_x['a'],deviceid_train['app_list']))
            filte1=np.logical_and(deviceid_train.n_class==i,a)
            print(np.unique(filte1))
            filte2=np.logical_and(a,True)
            if li in app_dict:
                app_dict[li]=min(app_dict[li],deviceid_train.ix[filte1,'app_list'].shape[0]/deviceid_train.ix[filte2,'app_list'].shape[0])
            else:
                app_dict[li]=deviceid_train.ix[filte1,'app_list'].shape[0]/deviceid_train.ix[filte2,'app_list'].shape[0]
    deviceid_packages['app2_w']=deviceid_packages['app_list'].apply(lambda x:app_dict_get(x,app_dict))
    return  deviceid_packages.ix[:,['device_id','app2_w']]


def compute_date():
    import multiprocessing

    pool = multiprocessing.Pool(processes=3)
    deviceid_train=dev_id_train()
    deviceid_packages= pd.read_csv(FLAGS.file_path +'deviceid_packages.csv',)
    def app_list(text):
        app_list=text.split('|')
    #        print (app_list)
        return app_list
    deviceid_packages['app_list']=deviceid_packages['add_id_list'].apply(lambda line:app_list(line)).tolist()
    deviceid_train=pd.merge(deviceid_train,deviceid_packages,on=['device_id'],how='left') 
    result = []
    result.append(pool.apply_async(app1_w, (deviceid_packages,deviceid_train )))
    result.append(pool.apply_async(app2_w, (deviceid_packages,deviceid_train )))
    result.append(pool.apply_async(app3_w, (deviceid_packages,deviceid_train )))
    pool.close()
    pool.join()
        
    deviceid_packages=pd.merge(result[0].get(),result[1].get(),on=['device_id'],how='left')
    deviceid_packages=pd.merge(deviceid_packages,result[2].get(),on=['device_id'],how='left') 
    deviceid_packages=deviceid_packages.fillna(-1)
    print(deviceid_packages.head(5))
    
    deviceid_packages.to_csv(file_path+'09_deviceid_packages.csv',index= False)
    
    
if __name__=='__main__':
    start_time=time.time()

    compute_date()

# id,
    end_time=time.time()
    print('耗时:',end_time-start_time)


