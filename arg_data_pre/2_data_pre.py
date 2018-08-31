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
from flags import FLAGS, unparsed
reload(sys)
sys.setdefaultencoding('utf8')


def file_exists(filename):
    if not os.path.exists(filename):
        os.system(r"touch {}".format(filename))#调用系统命令行来创建文件

file_path=FLAGS.file_path


def truncate_table(table_name):
    sql='truncate table '+table_name
    return data_from_mysql(sql)

def data_from_mysql(sql):
    db.ping(reconnect=True)
    cursor.execute(sql)
    db.commit()
    ret=cursor.fetchall()
    ret=pd.DataFrame(line for line in ret)
    return ret

def define_n_class():
    sql='alter table package_label add n_class int(4) default 0'
    ret=data_from_mysql(sql)
    print(ret.head(3))
    
    sql='select * from package_label'
    package_label=data_from_mysql(sql)
    
    package_label['t1'].unique()
    
    package_label['n_class']=package_label.apply(lambda line:)
    
    print(ret.head(3))
    
if __name__=='__main__':
    start_time=time.time()
    sql='select * from package_label'
#    ret=data_from_mysql(sql)
#    print(ret.head(3))
    define_n_class()



# id,
    end_time=time.time()
    print('耗时:',end_time-start_time)



