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
files=os.listdir(file_path)
tsv_file_names=[]
csv_file_names=[]

for file in files:
    file_name=file_path+'/'+file
    csv_file_name=file.replace('tsv','csv')
    file_exists(file_name.replace('tsv','csv'))
    tsv_file_names.append(file_name)
    csv_file_names.append(csv_file_name)
csv_file_names=list(set(csv_file_names))
print(csv_file_names)

cwd='/home/zhijiehuang/github/sex_age/arg_data_pre'


def tsv_to_csv():
    
    p = Popen('pwd' ,stdin =  PIPE,stdout = PIPE, stderr = PIPE,shell=True)  
    print (p.stdout.read())
    
    cmd='echo device_id,brand,type_no > '+file_path+'deviceid_brand.txt'
    print(cmd) 
    p = Popen(cmd ,stdin =  PIPE,stdout = PIPE, stderr = PIPE,cwd=cwd,shell=True)  
    

    cmd='cat '+file_path+"deviceid_brand.tsv |awk '{print $1,\",\",$2,\",\",$3}' >> "+file_path+'deviceid_brand.txt'
    print(cmd) 
    p = Popen(cmd ,stdin =  PIPE,stdout = PIPE, stderr = PIPE,cwd=cwd,shell=True)  
    
    
    cmd='echo device_id,app_id,start,close > '+file_path+'deviceid_package_start_close.txt'
    print(cmd) 
    p = Popen(cmd,stdin =  PIPE,stdout = PIPE, stderr = PIPE,cwd=cwd,shell=True)    
    

    cmd='cat '+file_path+"deviceid_package_start_close.tsv |awk '{print $1,\",\",$2,\",\",$3,\",\",$4}' >> "+file_path+'deviceid_package_start_close.txt'
    print(cmd) 
    p = Popen(cmd,stdin =  PIPE,stdout = PIPE, stderr = PIPE,cwd=cwd,shell=True)  
    


    cmd='echo app_id,t1,t2 > '+file_path+'package_label.txt'
    print(cmd) 
    p = Popen(cmd,stdin =  PIPE,stdout = PIPE, stderr = PIPE,cwd=cwd,shell=True)    
    

    cmd='cat '+file_path+"package_label.tsv |awk '{print $1,\",\",$2,\",\",$3}' >> "+file_path+'package_label.txt'
    print(cmd) 
    p = Popen(cmd,stdin =  PIPE,stdout = PIPE, stderr = PIPE,cwd=cwd,shell=True)    
    
    
    cmd='echo device_id,add_id_list > '+file_path+'deviceid_packages.txt'
    print(cmd) 
    p = Popen(cmd,stdin =  PIPE,stdout = PIPE, stderr = PIPE,cwd=cwd,shell=True)    
    

    cmd='cat '+file_path+"deviceid_packages.tsv |awk '{print $1,\",\",$2}' >> "+file_path+'deviceid_packages.txt'
    print(cmd) 
    p = Popen(cmd,stdin =  PIPE,stdout = PIPE, stderr = PIPE,cwd=cwd,shell=True)    
    
    
    cmd='echo device_id > '+file_path+'deviceid_test.txt'
    print(cmd) 
    p = Popen(cmd,stdin =  PIPE,stdout = PIPE, stderr = PIPE,cwd=cwd,shell=True)    
    

    cmd='cat '+file_path+"deviceid_test.tsv |awk '{print $1}' >> "+file_path+'deviceid_test.txt'
    print(cmd) 
    p = Popen(cmd,stdin =  PIPE,stdout = PIPE, stderr = PIPE,cwd=cwd,shell=True)    
    
    
    cmd='echo device_id > '+file_path+'deviceid_train.txt'
    print(cmd) 
    p = Popen(cmd,stdin =  PIPE,stdout = PIPE, stderr = PIPE,cwd=cwd,shell=True)    
    

    cmd='cat '+file_path+"deviceid_train.tsv |awk '{print $1}' >> "+file_path+'deviceid_train.txt'
    print(cmd) 
    p = Popen(cmd,stdin =  PIPE,stdout = PIPE, stderr = PIPE,cwd=cwd,shell=True)    
       
    
    
    


def pre_deviceid_packages():
    all_list=[]
    with open(file_path+'deviceid_packages.tsv','r') as fh:
        for line in fh: 
            
            line_list=line.replace('\t',',').replace(' ',',').replace('\n','').split(',')
            print(line_list)
            id_dict={}
            id_no=line_list[0]
            app_list=line_list[1:]
            id_dict['device_id']=id_no
            id_dict['add_id_list']='|'.join(app_list)
    #        for i,x in enumerate(line_list):
    #            id_dict['app_'+str(i)]=x
            
            all_list.append(id_dict)
           
    data_src=pd.DataFrame(x for x in all_list)
    #data_src=pd.read_csv(file_path+'package_label.csv')
    
    print(data_src.head(5))
    data_src.to_csv(file_path+'deviceid_packages.csv')

def pre_data(file_name):
    all_list=[]
    file_name=file_name.replace('.csv','')
    with open(file_path+file_name+'.txt','r') as fh:
        col=fh.readline()
        col_list=col.replace('\t',',').replace(' ',',').replace('\n','').split(',')
        for line in fh: 
            line_list=line.replace('\t',',').replace(' ','').replace('\n','').split(',')
            print(line_list)
            id_dict={}
            for i,x in enumerate(line_list):
                id_dict[col_list[i]]=x
            
            all_list.append(id_dict)
           
    data_src=pd.DataFrame(x for x in all_list)
    
    print(data_src.head(5))
    data_src.to_csv(file_path+'deviceid_packages.csv')

def data_into_mysql(file_name):
    data_src=pd.read_csv(file_path+file_name)
    table_name=file_name.replace('.csv','')
    print(table_name)
    print(data_src.head(5))
    pd.io.sql.to_sql(data_src,table_name, engine,if_exists='append', index= False)
    

if __name__=='__main__':
#1、
    tsv_to_csv()
    pre_deviceid_packages()
#2、
    for file_name in csv_file_names:
        pre_data(file_name)
#        data_into_mysql(file_name)
   
#3、
#    for file_name in csv_file_names:
#        data_into_mysql(file_name)
# id,



