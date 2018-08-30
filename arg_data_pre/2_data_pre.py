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


from sqlalchemy import create_engine


sys.path.append('..')

engine = create_engine('mysql+pymysql://root:root@localhost:3306/sex_age')

def file_exists(filename):
    if not os.path.exists(filename):
        os.system(r"touch {}".format(filename))#调用系统命令行来创建文件

file_path='/home/zhijiehuang/github/data/sex_age/'


def data_from_mysql(sql):
# 执行SQL
# cur = engine.execute('select * from hosts')
# 获取第一行数据
# cur.fetchone()
# 获取第n行数据
# cur.fetchmany(3)
# 获取所有数据
# cur.fetchall()
    cur = engine.execute(sql)
    ret=cur.fetchall()
    return ret

if __name__=='__main__':
    start_time=time.time()
    sql='select * from package_label'
    ret=data_from_mysql(sql)
    print(ret)


# id,
    end_time=time.time()
    print('耗时:',end_time-start_time)



