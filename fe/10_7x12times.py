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
import operator 

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.DEBUG)



file_path=FLAGS.file_path

def deviceid_package_start_close_train(deviceid):
    sql='select * from deviceid_package_start_close where device_id=\"'
    sql=sql+deviceid+'\" '
#    print(sql)
    ret=data_from_mysql(sql)
    return ret


def get_times_len(dev_id):
    ret_dict={}
    logging.debug(dev_id)
    ret=deviceid_package_start_close_train(dev_id,)
    if ret.shape[0]<1:
        return ret_dict
    gp=ret.groupby(['week','hour_bin'])['time_len']
    for (week,hour_bin),time_len in gp:
#        logging.debug(time_len)
#        logging.debug(len(time_len))
        key=int(sum(time_len)/len(time_len)/10)*10
        key=min(100,max(10,key))
        ret_dict[week+'_'+hour_bin]=key
    # 获得时长最长 t1
    
    # t1 时长
    
    # 获得 week hour_bin 时长
    
    return ret_dict



def devid_times(deviceid_packages):

#    def app_list(text):
#        app_list=text.split('|')
#        return app_list
#    deviceid_packages['app_list']=deviceid_packages['app_id_list'].apply(lambda line:app_list(line)).tolist()
     
    deviceid_packages['times_len']=deviceid_packages.apply(lambda line:get_times_len(line['device_id']) ,axis=1)
    
    all_col=reduce(operator.add, deviceid_packages['times_len'].apply(lambda x:list(x.keys())).tolist())
    all_col=list(set(all_col))
    columns=['device_id',]
    for col in all_col:
        logging.debug(col)
        col=col.replace(' ','')
        columns.append(col)
        def c(a,b):
#            print(a,b)
            ert=(a in b.keys())
            return ert
        _x=[]
        for _ in range(deviceid_packages.shape[0]):
            _x.append(col)
    #        print(_x)
        _x=pd.DataFrame({'a':_x},dtype='category')
        a=list(map(c,_x['a'],deviceid_packages['times_len']))
        filte1=np.logical_and(a,True)
        def get_values(t1_dict):
#            print(t1_dict)
            return t1_dict[col]
        values=deviceid_packages.loc[filte1,'times_len'].apply(lambda x:get_values(x)) 
#        logging.debug(values)
        deviceid_packages.loc[filte1,col]=values
    
    return deviceid_packages.ix[:, columns]
    
 
    
def compute_date():
    import multiprocessing

#    pool = multiprocessing.Pool(processes=2)
    deviceid_packages=pd.read_csv(file_path+'deviceid_packages.csv')

#    result = []
#    result.append(pool.apply_async(devid_times, (deviceid_packages, )))
#    result.append(pool.apply_async(devid_app_times_tx, (deviceid_packages,package_label, )))
#    pool.close()
#    pool.join()
    deviceid_packages=devid_times(deviceid_packages)
    deviceid_packages=deviceid_packages.fillna('0')
#    deviceid_packages=pd.merge(result[0].get(),result[1].get(),on=['device_id'],how='left') 
    
    logging.debug(deviceid_packages.head(5))
    
    deviceid_packages.to_csv(file_path+'10_deviceid_packages.csv',index= False)
    
    
if __name__=='__main__':
    start_time=time.time()

    compute_date()

# id,
    end_time=time.time()
    print('耗时:',end_time-start_time)


