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
import scipy.spatial.distance as ssd

from collections import defaultdict
sys.path.append('..')
sys.path.append('../..')
from db.conn_db import db,cursor,engine,truncate_table,data_from_mysql,dev_id_train,get_package_label
from flags import FLAGS, unparsed
from functools import reduce

import time, datetime

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
    train_dict=deviceid_packages.loc[:,['device_id','add_list']].to_dict()
    for line in :
        cols = line.strip().split(b",")
        i = userIndex[cols[0]]  #用户
        j = eventIndex[cols[1]] #活动
        
        eventsForUser[i].add(j)    #该用户参加了这个活动
        usersForEvent[j].add(i)    #该活动被用户参加
            
        #userEventScores[i, j] = int(cols[4]) - int(cols[5])   #interested - not_interested
        score = int(cols[4])
        #if score == 0:  #0在稀疏矩阵中表示该元素不存在，因此借用-1表示interested=0
        #userEventScores[i, j] = -1
        #else:
        userEventScores[i, j] = score
