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

file_path=FLAGS.file_path

def dev_id_train():
    sql='select * from deviceid_train'
    deviceid_train=data_from_mysql(sql)
    return deviceid_train

def tx_group_by(tx_pd,col='t1'):
    
    _key_codes = tx_pd[col].values
    cnt1=tx_pd['app_id'].groupby(_key_codes).size()
#    cnt1 = grp1.aggregate(np.size)
    _cnt = cnt1[_key_codes].values
    _cnt[np.isnan(_cnt)] = 0
    tx_pd[col+'_size'] = _cnt


def app_get_tx(app_list):
    
    tx_list=[]
    for app_id in app_list:
        t1_dict={}
        t2=get_package_label(app_id,'t1,t2')
        print(t2)
        if t2.shape[0]<1:
            continue
        t1_dict['t1']=t2['t1'].values
        t1_dict['t2']=t2['t2'].values
        t1_dict['app_id']=app_id
        tx_list.append(t1_dict)
    tx_pd=pd.DataFrame(x for x in tx_list)
    tx_group_by(tx_pd,'t1')
    tx_group_by(tx_pd,'t2')
    result_t1={}
    for t1 in tx_pd.t1.unique():
        result_t1[t1]=tx_pd[tx_pd.t1.values==t1,:].groupby('t1')['t1_size'].sum()
    result_t2={}
    for t2 in tx_pd.t2.unique():
        result_t2[t2]=tx_pd[tx_pd.t2.values==t2,:].groupby('t2')['t2_size'].sum()
    return result_t1,result_t2


def devid_app_tx(deviceid_packages,package_label):
    
    def app_list(text):
        app_list=text.split('|')
#        print (app_list)
        return app_list
    deviceid_packages['add_list']=deviceid_packages['add_id_list'].apply(lambda line:app_list(line)).tolist()
    deviceid_packages['t1_app_len'],deviceid_packages['t2_app_len']=deviceid_packages['add_list'].apply(lambda line:app_get_tx(line))
    
    columns=[]
    for x in FLAGS.t1_feature.replaces('\'').plit(','):
        columns.append('app_len_t1_'+str(x))
    for x in FLAGS.t1_feature.replaces('\'').plit(','):
        columns.append('app_len_t2_'+str(x))
        
    for x in package_label['t1'].unique():
        deviceid_packages['app_len_t1_'+str(x)]=0

    for x in package_label['t1'].unique():
        deviceid_packages['app_len_t2_'+str(x)]=0
    
    for x in package_label['t1'].unique():
        _x=[]
        for i in range(deviceid_packages.shape[0]):
            _x.append(str(x))
        print(_x)
        _x=pd.DataFrame({'a':_x},dtype='category')

        def c(a,b):
            ert=(str(a) in b.keys())
#            print(ert)
            return ert
        a=list(map(c,_x['a'],deviceid_packages['t1_app_len']))


        filte=np.logical_and(a,True)
        def get_values(t1_dict):
            return t1_dict[str(x)]
            
        values=deviceid_packages.ix[filte,'t1_app_len'].apply(lambda x:get_values(x))
#        print(filte)
        deviceid_packages.ix[filte,'app_len_t1_'+str(x)]=values
        
        
    for x in package_label['t2'].unique():
        _x=[]
        for i in range(deviceid_packages.shape[0]):
            _x.append(str(x))
        print(_x)
        _x=pd.DataFrame({'a':_x},dtype='category')

        def c(a,b):
            ert=(str(a) in b.keys())
#            print(ert)
            return ert
        a=list(map(c,_x['a'],deviceid_packages['t2_app_len']))


        filte=np.logical_and(a,True)
            
        values=deviceid_packages.ix[filte,'t2_app_len'].apply(lambda x:get_values(x))
#        print(filte)
        deviceid_packages.ix[filte,'app_len_t2_'+str(x)]=values
    columns.append('device_id')
    
    return deviceid_packages.ix[:, columns]
    

def devid_app_count(deviceid_packages,package_label):


    def app_count(text):
        app_list=text.split('|')
        return len(app_list)
    deviceid_packages['app_len']=deviceid_packages['add_id_list'].apply(lambda line:app_count(line))

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
    
    return deviceid_packages.ix[:, ['device_id','app_len','t1_code','t2_code']]
    
def compute_date():
    import multiprocessing

    pool = multiprocessing.Pool(processes=2)
    deviceid_packages=pd.read_csv(file_path+'deviceid_packages.csv')[:50]
    
    package_label=pd.read_csv(file_path+'package_label.csv')
    package_label['t1']=package_label['t1'].astype('category').values.codes
    package_label['t2']=package_label['t2'].astype('category').values.codes
    result = []
    result.append(pool.apply_async(devid_app_count, (deviceid_packages,package_label, )))
    result.append(pool.apply_async(devid_app_tx, (deviceid_packages,package_label, )))
    pool.close()
    pool.join()
    for res in result:
        ret=res.get()
        print('============================================',ret.head(2))
        deviceid_packages=pd.merge(deviceid_packages,ret,on=['device_id'],how='left') 
    
    print(deviceid_packages.head(5))
    
    deviceid_packages.to_csv(file_path+'02_deviceid_packages.csv', columns=['device_id','app_len','t1_code','t2_code','t1_app_len','t1_app_len'],index= False)
    
    
if __name__=='__main__':
    start_time=time.time()

    compute_date()

# id,
    end_time=time.time()
    print('耗时:',end_time-start_time)



