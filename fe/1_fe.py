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
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix, hstack
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.DEBUG)



file_path=FLAGS.file_path

def brand_type_no_onehot(deviceid_packages):
    deviceid_packages.drop('device_id', axis=1,inplace = True)
    deviceid_packages['trainrow'] = np.arange(deviceid_packages.shape[0])
    brandencoder = LabelEncoder().fit(deviceid_packages.brand)
    deviceid_packages['brand'] = brandencoder.transform(deviceid_packages['brand'])
    Xtr_brand = csr_matrix((np.ones(deviceid_packages.shape[0]),
                           (deviceid_packages.trainrow, deviceid_packages.brand)))
    
    m = deviceid_packages.brand.str.cat(deviceid_packages.type_no)

    modelencoder = LabelEncoder().fit(m)
    deviceid_packages['type_no'] = modelencoder.transform(m)
    Xtr_type_no = csr_matrix((np.ones(deviceid_packages.shape[0]),
                           (deviceid_packages.trainrow, deviceid_packages.type_no)))
    
    Xtrain = hstack((Xtr_brand, Xtr_type_no), format='csr')
    ret=pd.DataFrame(Xtrain)
    
    return ret

    
        

def calcLeaveOneOut(df, vn,gby ):

    #每个特征取值对应的样本组成group
    _key_codes = df[gby].values
    logging.debug(_key_codes)
    grp1 = df[vn].groupby(_key_codes)


    #计算每个group中样本的先验均值、样本值之和、样本数目

    sum1 = grp1.aggregate(np.sum)
    cnt1 = grp1.aggregate(np.size)
    

    logging.debug(sum1)
    logging.debug(cnt1)

    _sum = sum1[_key_codes].values
    _cnt = cnt1[_key_codes].values




#    _cnt[np.isnan(_cnt)] = 0    
#    _sum[np.isnan(_sum)] = 0

#    logging.debug(_cnt[:10])
#    logging.debug(_sum[:10])





    vn_cnt=gby+'_cnt'
    df[vn_cnt] = _cnt
    columns=[vn_cnt]
    return  df.ix[:,columns]


def type_no_w(deviceid_packages):
    deviceid_train=dev_id_train()
    deviceid_train=pd.merge(deviceid_train,deviceid_packages,on=['device_id'],how='left') 
    deviceid_train=deviceid_train.fillna(-1)
    typeno_1=deviceid_train.ix[deviceid_train['sex'].values==1,'type_no'].tolist()
    typeno_2=deviceid_train.ix[deviceid_train['sex'].values==2,'type_no'].tolist()
    all_typeno=deviceid_packages['type_no'].tolist()
    typeno_1_typeno_2=list(set(typeno_1).difference(set(typeno_2)))
    typeno_2_typeno_1=list(set(typeno_2).difference(set(typeno_1)))
    typeno_1andtypeno_2=list(list(set(typeno_1).intersection(set(typeno_2))))
    typeno_1notypeno_2=list(set(all_typeno).difference(set(typeno_1_typeno_2+typeno_2_typeno_1+typeno_1andtypeno_2)))
    typeno_dict={}
    for x in typeno_1_typeno_2:
        filte1=np.logical_and(deviceid_train.sex==1,deviceid_train.type_no==x)
        filte2=np.logical_and(deviceid_train.type_no==x,True)
        typeno_dict[x]=1
    for x in typeno_2_typeno_1:
        filte1=np.logical_and(deviceid_train.sex==2,deviceid_train.type_no==x)
        filte2=np.logical_and(deviceid_train.type_no==x,True)
        typeno_dict[x]=2
    for x in typeno_1andtypeno_2:
        filte1=np.logical_and(deviceid_train.sex==1,deviceid_train.type_no==x)
        filte2=np.logical_and(deviceid_train.sex==2,deviceid_train.type_no==x)
        s1=deviceid_train.ix[filte1,'type_no'].shape[0]
        s2=deviceid_train.ix[filte2,'type_no'].shape[0]
        p1=s1/(s1+s2)
        p2=1-p1
        p=p1 * np.log(p2)+p2 * np.log(p1)
        if x in typeno_dict:
            typeno_dict[x]=min(typeno_dict[x],-p/2)
        else:
            typeno_dict[x]=-p/2
    for x in typeno_1notypeno_2:
        typeno_dict[x]=0
    deviceid_packages['type_no_w']=deviceid_packages['type_no'].apply(lambda x:typeno_dict[x]) 
    columns=['type_no_w']
    return  deviceid_packages.ix[:,columns]

def difference_list(type_list):
    import copy
    import operator  
    retlist=[]
    for i in range(0,len(type_list)):
        
        alist=type_list[i]
        tlist=copy.deepcopy(type_list)
        tlist.remove(alist)
        blist=reduce(operator.add, tlist)
        clist=list(set(alist).difference(set(blist)))
        retlist.append(clist)
    return retlist
        
def type_no_w2(deviceid_packages):
    deviceid_train=dev_id_train()
    deviceid_train=deviceid_train.fillna(-1)
    deviceid_train=pd.merge(deviceid_train,deviceid_packages,on=['device_id'],how='left') 
    type_list=[]
    all_typeno=deviceid_packages['type_no'].tolist()
    train_typeno=deviceid_train['type_no'].tolist()
    no_train_typeno=list(set(all_typeno).difference(set(train_typeno)))
    for i in range(0,11):
        type_list1=deviceid_train.ix[deviceid_train['age'].values==i,'type_no'].tolist()
        type_list.append(type_list1)
    
    diff_list=difference_list(type_list)
    typeno_dict={}
    for i,x in enumerate(diff_list):
        for li in x:
            filte1=np.logical_and(deviceid_train.age==i,deviceid_train.type_no==li)
            filte2=np.logical_and(True,deviceid_train.type_no==li)
            if deviceid_train.ix[filte1,'type_no'].shape[0]==0:
                typeno_dict[li]=0
            else:
                typeno_dict[li]=deviceid_train.ix[filte1,'type_no'].shape[0]/deviceid_train.ix[filte2,'type_no'].shape[0]
    
    
    for i in range(0,11):
        clist=list(set(type_list[i]).difference(set(diff_list[i]+no_train_typeno)))
        for x in clist:
            filte1=np.logical_and(deviceid_train.age==i,deviceid_train.type_no==x)
            filte2=np.logical_and(True,deviceid_train.type_no==x)
            if x in typeno_dict:
                typeno_dict[x]=min(typeno_dict[x],(deviceid_train.ix[filte1,'type_no'].shape[0]/deviceid_train.ix[filte2,'type_no'].shape[0]))
            else:
                typeno_dict[x]=deviceid_train.ix[filte1,'type_no'].shape[0]/deviceid_train.ix[filte2,'type_no'].shape[0]
    for x in no_train_typeno:
        typeno_dict[x]=0
    deviceid_packages['type_no2_w']=deviceid_packages['type_no'].apply(lambda x:typeno_dict[x])
    columns=['type_no2_w']
    return  deviceid_packages.ix[:,columns]
    
def type_no_w3(deviceid_packages):
    deviceid_train=dev_id_train()
    deviceid_train=deviceid_train.fillna(-1)
    deviceid_train=pd.merge(deviceid_train,deviceid_packages,on=['device_id'],how='left') 
    type_list=[]
    all_typeno=deviceid_packages.type_no.tolist()
    train_typeno=deviceid_train.type_no.tolist()
    no_train_typeno=list(set(all_typeno).difference(set(train_typeno)))
    for i in range(0,23):
        type_list1=deviceid_train.ix[deviceid_train['n_class'].values==i,'type_no'].tolist()
        type_list.append(type_list1)
    
    diff_list=difference_list(type_list)
    typeno_dict={}
    for i,x in enumerate(diff_list):
        for li in x:
            filte1=np.logical_and(deviceid_train.n_class==i,deviceid_train.type_no==li)
            filte2=np.logical_and(True,deviceid_train.type_no==li)
            if deviceid_train.ix[filte1,'type_no'].shape[0]==0:
                typeno_dict[li]=0
            else:
                typeno_dict[li]=deviceid_train.ix[filte1,'type_no'].shape[0]/deviceid_train.ix[filte2,'type_no'].shape[0]
    
    
    for i in range(0,23):
        clist=list(set(type_list[i]).difference(set(diff_list[i]+no_train_typeno)))
        for x in clist:
            filte1=np.logical_and(deviceid_train.n_class==i,deviceid_train.type_no==x)
            filte2=np.logical_and(True,deviceid_train.type_no==x)
            if x in typeno_dict:
                typeno_dict[x]=min(typeno_dict[x],(deviceid_train.ix[filte1,'type_no'].shape[0]/deviceid_train.ix[filte2,'type_no'].shape[0]))
            else:
                typeno_dict[x]=deviceid_train.ix[filte1,'type_no'].shape[0]/deviceid_train.ix[filte2,'type_no'].shape[0]
    for x in no_train_typeno:
        typeno_dict[x]=0
    deviceid_packages['type_no3_w']=deviceid_packages['type_no'].apply(lambda x:typeno_dict[x])
    columns=['type_no3_w']
    return  deviceid_packages.ix[:,columns]

def compute_date():
    import multiprocessing

    pool = multiprocessing.Pool(processes=6)
    deviceid_packages=pd.read_csv(file_path+'deviceid_brand.csv')
    
#    package_label=pd.read_csv(file_path+'package_label.csv')
    deviceid_packages['brand']=deviceid_packages['brand'].astype('category').values.codes
    deviceid_packages['type_no']=deviceid_packages['type_no'].astype('category').values.codes
    device_id=deviceid_packages.ix[:,'device_id']
    brand_type_no_onehot(deviceid_packages)
    calcLeaveOneOut(deviceid_packages,'device_id','brand')
    calcLeaveOneOut(deviceid_packages,'device_id','type_no')
    type_no_w(deviceid_packages)
    type_no_w2(deviceid_packages)
    type_no_w3(deviceid_packages)
    
    result = []
    result.append(pool.apply_async(brand_type_no_onehot, (deviceid_packages, )))
    result.append(pool.apply_async(calcLeaveOneOut, (deviceid_packages,'device_id','brand',)))
    result.append(pool.apply_async(calcLeaveOneOut, (deviceid_packages,'device_id','type_no',)))
    result.append(pool.apply_async(type_no_w, (deviceid_packages, )))
    result.append(pool.apply_async(type_no_w2, (deviceid_packages, )))
    result.append(pool.apply_async(type_no_w3, (deviceid_packages, )))
    pool.close()
    pool.join()
        
    deviceid_packages=pd.concat([device_id,result[0].get(),result[1].get(),result[2].get(), \
                                 result[3].get(),result[4].get(),result[5].get()],axis=1)

    
    print(deviceid_packages.head(5))
    
    deviceid_packages.to_csv(file_path+'01_deviceid_packages.csv',index= False)
    
    
if __name__=='__main__':
    start_time=time.time()

    compute_date()

# id,
    end_time=time.time()
    print('耗时:',end_time-start_time)


