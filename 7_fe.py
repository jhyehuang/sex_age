# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 16:52:13 2018

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
#相似度/距离

import pickle as  pk

import itertools

#处理事件字符串
import datetime

import numpy as np
import scipy.io as sio
import scipy.sparse as ss
import scipy.spatial.distance as ssd

from collections import defaultdict
sys.path.append('..')
sys.path.append('../..')
from db.conn_db import db,cursor,engine,truncate_table,data_from_mysql,dev_id_train,get_package_label
from flags import FLAGS, unparsed
from functools import reduce

import time, datetime

#读取训练数据
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.decomposition import PCA



logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.DEBUG)

file_path=FLAGS.file_path

def deviceid_app():
    
    deviceid_packages=pd.read_csv(file_path+'deviceid_packages.csv')
    
    def app_list(text):
        app_list=text.split('|')
#        print (app_list)
        return app_list
    deviceid_packages['add_list']=deviceid_packages['add_id_list'].apply(lambda line:app_list(line)).tolist()

    # 统计训练集中有多少不同的用户的events
    unique_deviceid = set(deviceid_packages['device_id'].values.tolist())
    unique_app = set(deviceid_packages['add_list'].values.tolist())
    
    n_unique_deviceid = len(unique_deviceid)
    n_unique_app = len(unique_app)
    
    print("number of uniqueUsers :%d" % n_unique_deviceid)
    print("number of uniqueEvents :%d" % n_unique_app)
    
    #用户关系矩阵表，可用于后续LFM/SVD++处理的输入
    #这是一个稀疏矩阵，记录用户对活动感兴趣
    dev_app_Scores = ss.dok_matrix((n_unique_deviceid, n_unique_app))
    dev_Index = dict()
    app_Index = dict()
    
    #重新编码用户索引字典
    for i, u in enumerate(unique_deviceid):
        dev_Index[u] = i
        
    #重新编码活动索引字典    
    for i, e in enumerate(unique_app):
        app_Index[e] = i
        
        
    #统计每个用户参加的活动   / 每个活动参加的用户
    appFordev = defaultdict(set)
    devForapp = defaultdict(set)
    
    
    n_records = 0
    train_dict=deviceid_packages.loc[:,['device_id','add_list']].to_dict(orient='records')
    for line in train_dict:
        device_id = line.get('device_id','')
        app_list = line.get('add_list','')
        for app in app_list:
            i = dev_Index[device_id]  #用户
            j = app_Index[app_list] #活动
        
            appFordev[i].add(j)    #该用户参加了这个活动
            devForapp[j].add(i)    #该活动被用户参加
            dev_app_Scores[i, j] = 1
            
  
    ##统计每个用户参加的活动，后续用于将用户朋友参加的活动影响到用户
    pk.dump(devForapp, open("devForapp.pkl", 'wb'))
    ##统计活动参加的用户
    pk.dump(appFordev, open("appFordev.pkl", 'wb'))
    
    #保存用户-活动关系矩阵R，以备后用
    sio.mmwrite("dev_app_Scores", dev_app_Scores)
    
    
    #保存用户索引表
    pk.dump(dev_Index, open("dev_Index.pkl", 'wb'))
    #保存活动索引表
    pk.dump(app_Index, open("app_Index.pkl", 'wb'))


    # 为了防止不必要的计算，我们找出来所有关联的用户 或者 关联的event
    # 所谓的关联用户，指的是至少在同一个event上有行为的用户pair
    # 关联的event指的是至少同一个user有行为的event pair
    unique_dev_Pairs = set()
    unique_app_Pairs = set()
    for deviceid in unique_deviceid:
        i = dev_Index[deviceid]
        dev_id = devForapp[i]
        if len(dev_id) > 2:
            unique_dev_Pairs.update(itertools.combinations(dev_id, 2))
    
    for user in n_unique_app:
        u = app_Index[user]
        apps = appFordev[u]
        if len(apps) > 2:
            unique_app_Pairs.update(itertools.combinations(apps, 2))
     
    #保存用户-事件关系对索引表
    pk.dump(unique_dev_Pairs, open("unique_dev_Pairs.pkl", 'wb'))
    pk.dump(unique_app_Pairs, open("unique_app_Pairs.pkl", 'wb'))
    
    
def time_profile(func):
    def warpper(*args,**kwargs):
        start=time.time()
        result=func(*args,**kwargs)
        end=time.time()
        print('花费时间为{}'.format(end-start))
        return result
    return warpper
    