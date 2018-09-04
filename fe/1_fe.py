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
from db.conn_db import db,cursor,engine,truncate_table,data_from_mysql,get_package_dict
from flags import FLAGS, unparsed
from functools import reduce
import logging

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.DEBUG)



file_path=FLAGS.file_path

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





    vn_cnt=vn+'_cnt'
    df[vn_cnt] = _cnt


    
def compute_date():
    import multiprocessing

#    pool = multiprocessing.Pool(processes=2)
    deviceid_packages=pd.read_csv(file_path+'deviceid_brand.csv')
    
#    package_label=pd.read_csv(file_path+'package_label.csv')
    deviceid_packages['brand']=deviceid_packages['brand'].astype('category').values.codes
    deviceid_packages['type_no']=deviceid_packages['type_no'].astype('category').values.codes
    
    calcLeaveOneOut(deviceid_packages,'device_id','brand')
    calcLeaveOneOut(deviceid_packages,'device_id','type_no')
    
#    result = []
#    result.append(pool.apply_async(devid_times, (deviceid_packages, )))
#    result.append(pool.apply_async(devid_app_times_tx, (deviceid_packages,package_label, )))
#    pool.close()
#    pool.join()
        
#    deviceid_packages=pd.merge(result[0].get(),result[1].get(),on=['device_id'],how='left') 
    
    print(deviceid_packages.head(5))
    
    deviceid_packages.to_csv(file_path+'01_deviceid_packages.csv',index= False)
    
    
if __name__=='__main__':
    start_time=time.time()

    compute_date()

# id,
    end_time=time.time()
    print('耗时:',end_time-start_time)


