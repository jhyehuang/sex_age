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
from db.conn_db import db,cursor,engine
from flags import FLAGS, unparsed



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
#    sql='alter table package_label add n_class int(4) default 0'
#    ret=data_from_mysql(sql)
#    print(ret.head(3))
    
    sql='select * from package_label'
    package_label=data_from_mysql(sql)
    
    sex=package_label['t1'].unique()
    age=package_label['t2'].unique()
    label=[]
    for i in sex:
        for j in age:
            label.append(str(i)+'-'+str(j))
    label_dict={}
    for i,x in enumerate(label):
        label_dict[x]=i
    label_pd=pd.DataFrame(x for x in [label_dict])
    label_pd.to_csv(file_path+'label.csv')
    def map_label(t1,t2):
        return label_dict[str(t1)+'-'+str(t2)]
        
    print(package_label.head(3))
    
    package_label['n_class']=package_label.apply(lambda line:map_label(line['t1'],line['t2']),axis=1)
    truncate_table('package_label')
    
    pd.io.sql.to_sql(package_label,'package_label', engine,if_exists='append', index= False)
    
    
if __name__=='__main__':
    start_time=time.time()
    sql='select * from package_label'
#    ret=data_from_mysql(sql)
#    print(ret.head(3))
    define_n_class()

# id,
    end_time=time.time()
    print('耗时:',end_time-start_time)



