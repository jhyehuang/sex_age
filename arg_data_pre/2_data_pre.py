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
sys.path.append('..')
from subprocess import *
from data_preprocessing import *

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

def data_into_mysql(file_name='deviceid_train.csv'):
    data_src=pd.read_csv(file_path+file_name)
    table_name=file_name.replace('.csv','')
    print(table_name)
    print(data_src.head(5))
    pd.io.sql.to_sql(data_src,table_name, engine,if_exists='append', index= False)

def define_n_class():
#    sql='alter table deviceid_train add n_class int(4) default 0'
#    ret=data_from_mysql(sql)
#    print(ret.head(3))
    
    sql='select * from deviceid_train'
    package_label=data_from_mysql(sql)
    
#    sex=package_label['sex'].unique()
#    age=package_label['age'].unique()
    label=[]
    for i in [1,2]:
        for j in range(11):
            label.append(str(i)+'-'+str(j))
    label_dict={}
    for i,x in enumerate(label):
        label_dict[x]=i
    label_pd=pd.DataFrame(x for x in [label_dict])
    label_pd.to_csv(file_path+'label.csv')
    def map_label(t1,t2):
        return label_dict[str(t1)+'-'+str(t2)]
        
    print(package_label.head(3))
    
    package_label['n_class']=package_label.apply(lambda line:map_label(line['sex'],line['age']),axis=1)
#    package_label=data_augmentation(package_label)
    print(package_label.shape)
    truncate_table('deviceid_train')
    package_label.to_csv(file_path+'package_label.csv',index= False)
    pd.io.sql.to_sql(package_label,'deviceid_train', engine,if_exists='append', index= False)
    
    
if __name__=='__main__':
    start_time=time.time()
    sql='select * from package_label'
#    ret=data_from_mysql(sql)
#    print(ret.head(3))
    truncate_table('deviceid_train')
    data_into_mysql()
    define_n_class()

# id,
    end_time=time.time()
    print('耗时:',end_time-start_time)



