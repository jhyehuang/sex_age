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


file_path=FLAGS.file_path
def compute_date():

    deviceid_packages=pd.read_csv(file_path+'deviceid_packages.csv')
#    deviceid_train=dev_id_train()
    deviceid_id=deviceid_packages.ix[:,['device_id','n_class']]

    logging.debug('============================================')
    
    logging.debug(deviceid_id.head(5))
    
    deviceid_id.to_csv(file_path+'device_id.csv',index= False)
    
if __name__=='__main__':
    start_time=time.time()

    compute_date()

# id,
    end_time=time.time()
    logging.debug('耗时:'+str(end_time-start_time))



