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
import logging
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

'''
# 初始化数据库连接，使用pymysql模块 # MySQL的用户：root, 
密码:root, 端口：3306,
数据库：
mydb 
'''

sys.path.append('..')

from db.conn_db import db,cursor,engine,truncate_table,data_from_mysql,get_package_label, \
    get_package_dict
from flags import FLAGS, unparsed
from functools import reduce

file_path=FLAGS.file_path

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.DEBUG)



def dev_id_train():
    sql='select * from deviceid_train'
    deviceid_train=data_from_mysql(sql)
    return deviceid_train

def word_to_tfidf(word):
    transformer=TfidfVectorizer()
    tfidf=transformer.fit_transform(word)
    weight=np.sum(tfidf.toarray(),axis=1).reshape((-1,1))
    return weight

def word_to_lda(word):
    vectorizer=CountVectorizer()
    tf = vectorizer.fit_transform(word)
    lda = LatentDirichletAllocation(n_topics=5,
                                    max_iter=50,
                                    learning_method='batch')
    docres = lda.fit_transform(tf)
    logging.debug(docres)
    lda_pd=pd.DataFrame(data=np.array(docres),columns=['app_lda_t2_'+str(i) for i in range(1,6)])
    logging.debug(lda_pd)
    return lda_pd

def tx_group_by(tx_pd,col='t1'):
    
    _key_codes = tx_pd[col].values
#    print(_key_codes)
    cnt1=tx_pd['app_id'].groupby(_key_codes).size()
#    cnt1 = grp1.aggregate(np.size)
    _cnt = cnt1[_key_codes].values
    _cnt[np.isnan(_cnt)] = 0
    tx_pd[col+'_size'] = _cnt

def app_list(text):
    app_list=text.split('|')
#        print (app_list)
    return app_list
    
def app_get_t1(app_list):
    logging.debug(app_list)
    if len(app_list)<1:
        return {}
    tx_list=[]
    for app_id in app_list:
        t1_dict={}
        t2=get_package_dict(app_id,'t1,t2')
        
        if len(t2)<1:
#            print(t2)
            continue
        t2=t2[0]
        t1_dict['t1']=t2.get('t1','0')
        t1_dict['t2']=t2.get('t2','0')
        t1_dict['app_id']=app_id
        tx_list.append(t1_dict)
    tx_pd=pd.DataFrame(x for x in tx_list)
    if tx_pd.shape[0]<1:
        return {}
    tx_group_by(tx_pd,'t1')

#    print(tx_pd)
    result_t1={}
    for t1 in tx_pd.t1.unique():
        result_t1[t1]=tx_pd.ix[tx_pd.t1.values==t1,'t1_size'].sum()
    logging.debug(result_t1)
    return result_t1


def app_get_t2(app_list):
    logging.debug(app_list)
    if len(app_list)<1:
        return {}
    tx_list=[]
    for app_id in app_list:
        t1_dict={}
        t2=get_package_dict(app_id,'t1,t2')
        if len(t2)<1:
#            print(t2)
            continue
        t2=t2[0]
        t1_dict['t1']=t2.get('t1','0')
        t1_dict['t2']=t2.get('t2','0')
        t1_dict['app_id']=app_id
        tx_list.append(t1_dict)
    tx_pd=pd.DataFrame(x for x in tx_list)
    if tx_pd.shape[0]<1:
        return {}
    tx_group_by(tx_pd,'t2')
#    print(tx_pd)

    result_t2={}
    for t2 in tx_pd.t2.unique():
        result_t2[t2]=tx_pd.ix[tx_pd.t2.values==t2,'t2_size'].sum()
    logging.debug(result_t2)
    return result_t2


def devid_app_tx(deviceid_packages,package_label):
    
    deviceid_packages['add_list']=deviceid_packages['add_id_list'].apply(lambda line:app_list(line)).tolist()
    deviceid_packages['t1_app_len']=deviceid_packages['add_list'].apply(lambda line:app_get_t1(line))
    deviceid_packages['t2_app_len']=deviceid_packages['add_list'].apply(lambda line:app_get_t2(line))
    
    columns=[]
    logging.debug(FLAGS.t1_feature.replace('\'','').split(','))
    for x in FLAGS.t1_feature.replace('\'','').split(','):
        columns.append('app_len_t1_'+str(x))
    for x in FLAGS.t2_feature.replace('\'','').split(','):
        columns.append('app_len_t2_'+str(x))
        
    for x in package_label['t1'].unique():
        deviceid_packages['app_len_t1_'+str(x)]=int(0)

    for x in package_label['t2'].unique():
        deviceid_packages['app_len_t2_'+str(x)]=int(0)
    
    for x in package_label['t1'].unique():
        _x=[]
        for i in range(deviceid_packages.shape[0]):
            _x.append(str(x))
#        print(_x)
        _x=pd.DataFrame({'a':_x},dtype='category')

        def c(a,b):
#            print(a,b)
            ert=(str(a) in b.keys())
#            print(ert)
            return ert
        a=list(map(c,_x['a'],deviceid_packages['t1_app_len']))


        filte=np.logical_and(a,True)
        def get_values(t1_dict):
            return t1_dict[str(x)]
            
        values=deviceid_packages.ix[filte,'t1_app_len'].apply(lambda x:get_values(x))
#        print(filte)
        deviceid_packages.ix[filte,'app_len_t1_'+str(x)]=values
        
        
    for x in package_label['t2'].unique():
        _x=[]
        for i in range(deviceid_packages.shape[0]):
            _x.append(str(x))
#        print(_x)
        _x=pd.DataFrame({'a':_x},dtype='category')

        def c(a,b):
            ert=(str(a) in b.keys())
#            print(ert)
            return ert
        a=list(map(c,_x['a'],deviceid_packages['t2_app_len']))


        filte=np.logical_and(a,True)
            
        values=deviceid_packages.ix[filte,'t2_app_len'].apply(lambda x:get_values(x))
#        print(filte)
        deviceid_packages.ix[filte,'app_len_t2_'+str(x)]=values
    columns.append('device_id')
    logging.debug(columns)
    
    return deviceid_packages.ix[:, columns]
    

def devid_app_count(deviceid_packages,package_label):

    def app_count(text):
        app_list=text.split('|')
        return len(app_list)
    deviceid_packages['app_len']=deviceid_packages['add_id_list'].apply(lambda line:app_count(line))


    app_mtrix=deviceid_packages['add_id_list'].apply(lambda line:app_list(line)).tolist()
#    app_mtrix=app_mtrix['add_id_list']
#    print(package_label['t1'].max())
#    print(package_label['t2'].max())
    
    def get_label_t1_1(l):
#        print(l)
        logging.debug(l)
        ret=list(map(get_label_2_t1,l))
        condition = lambda t: t != ""
        ret= list(filter(condition, ret))
        ret=list(map(str,ret))
        if len(ret)<1:
            ret.append('0')
        return ''.join(ret)
  
    def get_label_t2_1(l):
#        print(l)
        logging.debug(l)
        ret=list(map(get_label_2_t2,l))
        condition = lambda t: t != ""
        ret= list(filter(condition, ret))
        ret=list(map(str,ret))
        if len(ret)<1:
            ret.append('0')
        return ''.join(ret)
  
    def get_label_2_t1(l):
        
        filer=package_label['app_id'].astype('category').values==l
        label=package_label.ix[filer,'t1'].values.tolist()
        if len(label)<1:
            return ''
        return label.pop()
    
    def get_label_2_t2(l):
        filer=package_label['app_id'].astype('category').values==l
        label=package_label.ix[filer,'t2'].values.tolist()
        if len(label)<1:
            return ''
        return label.pop()
    t1_mtrix=list(map(get_label_t1_1,app_mtrix))
    t2_mtrix=list(map(get_label_t2_1,app_mtrix))
    
    
    deviceid_packages['t1_code']=np.array(t1_mtrix).reshape(-1,1)
    deviceid_packages['t2_code']=np.array(t2_mtrix).reshape(-1,1)
    
    deviceid_packages['t1_code']=deviceid_packages['t1_code'].astype('category').values.codes
    deviceid_packages['t2_code']=deviceid_packages['t2_code'].astype('category').values.codes
    
#    deviceid_train=pd.merge(deviceid_train,deviceid_packages,on=['device_id'],how='left') 
    
 
    
#    print(deviceid_train.head(5))
    
    return deviceid_packages.ix[:, ['device_id','app_len','t1_code','t2_code']]
    
def devid_app_tfidf(deviceid_packages,package_label):
    def list_to_text(l):
        text=' '.join(l)
    #        print (app_list)
        return text

    deviceid_packages['add_id_text']=deviceid_packages['add_id_list'].apply(lambda line:list_to_text(app_list(line)))
    word=deviceid_packages['add_id_text'].values.tolist()    
    deviceid_packages['app_id_weight']=word_to_tfidf(word)

    
    def get_label_t1_1(l):
#        print(l)
        logging.debug(l)
        ret=list(map(get_label_2_t1,l))
        condition = lambda t: t != ""
        ret= list(filter(condition, ret))
        ret=list(map(str,ret))
        if len(ret)<1:
            ret.append('0')
        return ' '.join(ret)
  
    def get_label_t2_1(l):
#        print(l)
        logging.debug(l)
        ret=list(map(get_label_2_t2,l))
        condition = lambda t: t != ""
        ret= list(filter(condition, ret))
        ret=list(map(str,ret))
        if len(ret)<1:
            ret.append('0')
        return ' '.join(ret)
  
    def get_label_2_t1(l):
        
        filer=package_label['app_id'].astype('category').values==l
        label=package_label.ix[filer,'t1'].values.tolist()
        if len(label)<1:
            return ''
        return label.pop()
    
    def get_label_2_t2(l):
        filer=package_label['app_id'].astype('category').values==l
        label=package_label.ix[filer,'t2'].values.tolist()
        if len(label)<1:
            return ''
        return label.pop()
    app_mtrix=deviceid_packages['add_id_list'].apply(lambda line:app_list(line)).tolist()
    t1_mtrix=list(map(get_label_t1_1,app_mtrix))
    t2_mtrix=list(map(get_label_t2_1,app_mtrix))
    deviceid_packages['app_t1_weight']=word_to_tfidf(t1_mtrix)
    deviceid_packages['app_t2_weight']=word_to_tfidf(t2_mtrix)
    lda_pd=word_to_lda(t2_mtrix)
    logging.debug(lda_pd)
    deviceid_packages=pd.concat([deviceid_packages,lda_pd],axis=1)
    

    
    return deviceid_packages.ix[:, ['device_id','app_id_weight','app_t1_weight','app_t2_weight']+['app_lda_t2_'+str(i) for i in range(1,6)]]


def devid_app_brand_tfidf(deviceid_packages,deviceid_brand):
    def list_to_text(l):
        text=' '.join(l)
    #        print (app_list)
        return text
    
  
    def get_brand(l):
        
        filer=deviceid_brand['device_id'].astype('category').values==l
        label=deviceid_brand.ix[filer,'brand'].values.tolist()
        if len(label)<1:
            return ''
        return label.pop()
    
    def get_type_no(l):
        filer=deviceid_brand['device_id'].astype('category').values==l
        label=deviceid_brand.ix[filer,'type_no'].values.tolist()
        if len(label)<1:
            return ''
        return label.pop()
    app_mtrix=deviceid_packages['device_id'].tolist()
    t1_mtrix=list(map(get_brand,app_mtrix))
    t2_mtrix=list(map(get_type_no,app_mtrix))
    deviceid_packages['dev_brand_weight']=word_to_tfidf(t1_mtrix)
    deviceid_packages['dev_type_no_weight']=word_to_tfidf(t2_mtrix)

    
    return deviceid_packages.ix[:, ['device_id','dev_brand_weight','dev_type_no_weight']]


def compute_date():
    import multiprocessing

    pool = multiprocessing.Pool(processes=4)
    deviceid_packages=pd.read_csv(file_path+'deviceid_packages.csv')[50:100]
    deviceid_brand=pd.read_csv(file_path+'deviceid_brand.csv')
    
    package_label=pd.read_csv(file_path+'package_label.csv')
    package_label['t1']=package_label['t1'].astype('category').values.codes
    package_label['t2']=package_label['t2'].astype('category').values.codes
    result = []
    result.append(pool.apply_async(devid_app_count, (deviceid_packages,package_label, )))
    result.append(pool.apply_async(devid_app_tx, (deviceid_packages,package_label, )))
    result.append(pool.apply_async(devid_app_tfidf, (deviceid_packages,package_label, )))
    result.append(pool.apply_async(devid_app_brand_tfidf, (deviceid_packages,deviceid_brand, )))
    pool.close()
    pool.join()
        
    deviceid_packages=pd.merge(result[0].get(),result[1].get(),on=['device_id'],how='left')
    deviceid_packages=pd.merge(deviceid_packages,result[2].get(),on=['device_id'],how='left')
    deviceid_packages=pd.merge(deviceid_packages,result[3].get(),on=['device_id'],how='left')
    deviceid_packages.fillna(0)
    print(deviceid_packages.head(5))
    
    deviceid_packages.to_csv(file_path+'02_deviceid_packages.csv',index= False)
    
    
if __name__=='__main__':
    start_time=time.time()

    compute_date()

# id,
    end_time=time.time()
    print('耗时:',end_time-start_time)



