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
from data_preprocessing import *
import logging

def file_exists(filename):
    if not os.path.exists(filename):
        os.system(r"touch {}".format(filename))#调用系统命令行来创建文件

file_path=FLAGS.file_path

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.DEBUG)


def augmentatio_deviceid_packages():
#    sql='alter table deviceid_train add n_class int(4) default 0'
#    ret=data_from_mysql(sql)
#    print(ret.head(3))
    sql='select * from deviceid_train'
    package_label=data_from_mysql(sql)
    deviceid_packages=pd.read_csv(file_path+'deviceid_packages.csv')
    cols=deviceid_packages.columns
    deviceid_packages=pd.merge(deviceid_packages,package_label,on=['device_id'],how='left')
    deviceid_packages=data_augmentation(deviceid_packages)
    logging.debug(deviceid_packages.shape)
    deviceid_packages.to_csv(file_path+'deviceid_packages.csv',columns=cols,index= False)
    
    
if __name__=='__main__':
    start_time=time.time()

#    ret=data_from_mysql(sql)
#    print(ret.head(3))
    augmentatio_deviceid_packages()
    
# id,
    end_time=time.time()
    print('耗时:',end_time-start_time)



