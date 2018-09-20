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



def file_exists(filename):
    if not os.path.exists(filename):
        os.system(r"touch {}".format(filename))#调用系统命令行来创建文件

file_path=FLAGS.file_path


def hot_package_label():
#    sql='alter table deviceid_train add n_class int(4) default 0'
#    ret=data_from_mysql(sql)
#    print(ret.head(3))
    package_label=pd.read_csv(file_path+'package_label.txt')
    package_label['t1']=package_label['t1'].astype('category').values.codes
    package_label['t2']=package_label['t2'].astype('category').values.codes
    print(len(package_label['t1'].unique()))
    print(len(package_label['t2'].unique()))
    package_label.to_csv(file_path+'package_label.csv',columns=['app_id','t1','t2'],index= False)
    
def package_label_into_mysql():
    data_src=pd.read_csv(file_path+'package_label.csv')
    table_name='package_label'
    truncate_table(table_name)
    print(table_name)
    print(data_src.head(5))
    pd.io.sql.to_sql(data_src,table_name, engine,if_exists='append', index= False)
    
if __name__=='__main__':
    start_time=time.time()

#    ret=data_from_mysql(sql)
#    print(ret.head(3))
    hot_package_label()
#    package_label_into_mysql()
    
# id,
    end_time=time.time()
    print('耗时:',end_time-start_time)



