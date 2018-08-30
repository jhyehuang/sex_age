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
from db.conn_db import db,cursor



def file_exists(filename):
    if not os.path.exists(filename):
        os.system(r"touch {}".format(filename))#调用系统命令行来创建文件

file_path='/home/zhijiehuang/github/data/sex_age/'


def data_from_mysql(sql):
    db.ping(reconnect=True)
    cursor.execute(sql)
    db.commit()
    ret=cursor.fetchall()
    return ret

def define_n_class():
    sql='alter table package_label add n_class int(4) default 0'
    ret=data_from_mysql(sql)
    print(ret)
    
    sql='select * from package_label'
    ret=data_from_mysql(sql)
    print(ret)
    
if __name__=='__main__':
    start_time=time.time()
    sql='select * from package_label'
    ret=data_from_mysql(sql)
    print(ret)



# id,
    end_time=time.time()
    print('耗时:',end_time-start_time)



