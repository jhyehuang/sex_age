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
    if len(word)<1:
        return 0
    elif len(word)==1:
       return [1]
    elif len(list(set(word)))==1:
       return [1 for x in range(len(word))]
#    logging.debug(word)
#    logging.debug(list(set(word)))
    transformer=TfidfVectorizer(min_df=1)

    tfidf=transformer.fit_transform(word,)
    weight=np.sum(tfidf.toarray(),axis=1).reshape((-1,1))
#    logging.debug(weight)
    return weight

def word_to_lda(word):
    logging.debug(len(word))
    if len(word)<1:
        return 0
    elif len(word)==1:
       return [1]
    elif len(list(set(word)))==1:
       return pd.DataFrame(data=np.zeros((len(word),5)),columns=['app_lda_t2_'+str(i) for i in range(1,6)])
    logging.debug(word)
    logging.debug(list(set(word)))
    vectorizer=CountVectorizer(min_df=1)
    tf = vectorizer.fit_transform(word)
    lda = LatentDirichletAllocation(n_topics=5,
                                    max_iter=50,
                                    learning_method='batch')
    docres = lda.fit_transform(tf)
#    logging.debug(docres)
    lda_pd=pd.DataFrame(data=np.array(docres),columns=['app_lda_t2_'+str(i) for i in range(1,6)])
#    logging.debug(lda_pd)
    return lda_pd

def tx_group_by(tx_pd,col='t1'):
    
    _key_codes = tx_pd[col].values
#    print(_key_codes)
    cnt1=tx_pd['app_id'].groupby(_key_codes).size()
#    cnt1 = grp1.aggregate(np.size)
#    _cnt = cnt1[tx_pd[col].unique().tolist()].values
#    _cnt[np.isnan(_cnt)] = 0
#    tx_pd[col+'_size'] = _cnt
    return cnt1

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
        # 查得 app_id  对应的 t1 编码 t2 编码
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
    
    #计算 dev_id 下 按照t1 分组后每个t1的数量
    cnt1=tx_group_by(tx_pd,'t1')

#    print(tx_pd)
    result_t1={}
    for t1 in tx_pd.t1.unique():
        result_t1[t1]=cnt1[t1]
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
    cnt1=tx_group_by(tx_pd,'t2')
#    print(tx_pd)

    result_t2={}
    for t2 in tx_pd.t2.unique():
        result_t2[t2]=cnt1[t2]
    logging.debug(result_t2)
    return result_t2


def devid_app_tx(deviceid_packages,package_label):
    
    deviceid_packages['add_list']=deviceid_packages['add_id_list'].apply(lambda line:app_list(line)).tolist()
    #每一个app_list中 属于t1类型的size
    deviceid_packages['t1_app_len']=deviceid_packages['add_list'].apply(lambda line:app_get_t1(line))
    #每一个app_list中 属于t2类型的size
    deviceid_packages['t2_app_len']=deviceid_packages['add_list'].apply(lambda line:app_get_t2(line))
    
    columns=[]
#    logging.debug(FLAGS.t1_feature.replace('\'','').split(','))
#    for x in FLAGS.t1_feature.replace('\'','').split(','):
#        columns.append('app_len_t1_'+str(x))
#    for x in FLAGS.t2_feature.replace('\'','').split(','):
#        columns.append('app_len_t2_'+str(x))
        
    # 将 deviceid_packages['t1_app_len'] 展开package_label['t1'].unique()个特征
    deviceid_packages['app_t1_w']=int(0)
    columns.append('app_t1_w')

    deviceid_packages['app_t2_w']=int(0)
    columns.append('app_t2_w')
    
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
#        def get_values(t1_dict):
#            return t1_dict[str(x)]
            
#        values=deviceid_packages.ix[filte,'t1_app_len'].apply(lambda x:get_values(x))
#        print(filte)
        deviceid_packages.ix[filte,'app_t1_w']=deviceid_packages.ix[filte,'app_t1_w']+deviceid_packages.ix[filte,'device_id'].shape[0]/deviceid_packages.shape[0]
        
        
    # 将 deviceid_packages['t2_app_len'] 展开package_label['t2'].unique()个特征
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
            
#        values=deviceid_packages.ix[filte,'t2_app_len'].apply(lambda x:get_values(x))
#        print(filte)
        deviceid_packages.ix[filte,'app_t2_w']= deviceid_packages.ix[filte,'app_t2_w']+deviceid_packages.ix[filte,'device_id'].shape[0]/deviceid_packages.shape[0]
    columns.append('device_id')
    logging.debug(columns)
    deviceid_packages.fillna(0)
    
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
    # 将app_list 包含的t1类型进行join编码
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
  
    # 将app_list 包含的t2类型进行join编码
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
    deviceid_packages.fillna(0)
    
#    deviceid_train=pd.merge(deviceid_train,deviceid_packages,on=['device_id'],how='left') 
    
 
    
#    print(deviceid_train.head(5))
    
    return deviceid_packages.ix[:, ['device_id','app_len','t1_code','t2_code']]
    
def devid_app_tfidf(deviceid_packages,package_label):
    def list_to_text(l):
        text=' '.join(l)
    #        print (app_list)
        return text

    #将app_list 转化为空格分割的字符串
    deviceid_packages['add_id_text']=deviceid_packages['add_id_list'].apply(lambda line:list_to_text(app_list(line)))
    # 将分割的字符串变为shape[:,1]的数组
    word=deviceid_packages['add_id_text'].values.tolist() 
    # 数组作为tf * idf的输入，获得词条出现的词频-逆向文件频率
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
#        logging.debug(l)
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
    
    # 转化为t1 字符串 和 t2 字符串
    t1_mtrix=list(map(get_label_t1_1,app_mtrix))
    t2_mtrix=list(map(get_label_t2_1,app_mtrix))
    
    # 计算t1 字符串 和 t2 字符串 的词频 逆向文件频率
    deviceid_packages['app_t1_weight']=word_to_tfidf(t1_mtrix)
    deviceid_packages['app_t2_weight']=word_to_tfidf(t2_mtrix)
    
    # 计算 t2 的主题概率
    logging.debug(t2_mtrix)
    lda_pd=word_to_lda(t2_mtrix)
#    logging.debug(lda_pd)
#    logging.debug(lda_pd['app_lda_t2_1'].values)
    for x in ['app_lda_t2_'+str(i) for i in range(1,6)]:
        deviceid_packages[x]=lda_pd[x].values

    deviceid_packages.fillna(0)
    

    
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
        return ' '.join(label)
    
    def get_type_no(l):
        filer=deviceid_brand['device_id'].astype('category').values==l
        label=deviceid_brand.ix[filer,'type_no'].values.tolist()
        if len(label)<1:
            return ''
        return ' '.join(label)
    app_mtrix=deviceid_packages['device_id'].tolist()
    t1_mtrix=list(map(get_brand,app_mtrix))
    t2_mtrix=list(map(get_type_no,app_mtrix))
    deviceid_packages['dev_brand_weight']=word_to_tfidf(t1_mtrix)
    deviceid_packages['dev_type_no_weight']=word_to_tfidf(t2_mtrix)
    deviceid_packages.fillna(0)

    
    return deviceid_packages.ix[:, ['device_id','dev_brand_weight','dev_type_no_weight']]


def compute_date():
    import multiprocessing

    pool = multiprocessing.Pool(processes=3)
    deviceid_packages=pd.read_csv(file_path+'deviceid_packages.csv')[:50]
    deviceid_brand=pd.read_csv(file_path+'deviceid_brand.csv')
    
    deviceid_brand['brand']=deviceid_brand['brand'].astype('category').values.codes
    deviceid_brand['type_no']=deviceid_brand['type_no'].astype('category').values.codes
    deviceid_brand['brand']=deviceid_brand['brand'].map(str)
    deviceid_brand['type_no']=deviceid_brand['type_no'].map(str)
    
    package_label=pd.read_csv(file_path+'package_label.csv')
    package_label['t1']=package_label['t1'].astype('category').values.codes
    package_label['t2']=package_label['t2'].astype('category').values.codes
    result = []
    result.append(pool.apply_async(devid_app_count, (deviceid_packages,package_label, )))
    result.append(pool.apply_async(devid_app_tx, (deviceid_packages,package_label, )))
    result.append(pool.apply_async(devid_app_tfidf, (deviceid_packages,package_label, )))
#    result.append(pool.apply_async(devid_app_brand_tfidf, (deviceid_packages,deviceid_brand, )))
    pool.close()
    pool.join()
        
    deviceid_packages=pd.merge(result[0].get(),result[1].get(),on=['device_id'],how='left')
    deviceid_packages=pd.merge(deviceid_packages,result[2].get(),on=['device_id'],how='left')
#    deviceid_packages=pd.merge(deviceid_packages,result[3].get(),on=['device_id'],how='left')
    deviceid_packages.fillna(0)
    print(deviceid_packages.head(5))
#    columns=['device_id','app_id_weight','app_len_t1_43', 'app_len_t2_132', 'app_len_t1_36', 'app_len_t2_94',
#             'app_len_t2_251', 'app_len_t1_33', 'app_len_t2_124', 'app_len_t1_32', 'app_len_t2_223', 'app_len_t2_83',
#             'app_len_t2_106', 'app_len_t2_61', 'app_len_t2_159', 'app_len_t1_17', 'app_len_t2_158',]
    
    deviceid_packages.to_csv(file_path+'02_deviceid_packages.csv',index= False)
    
    
if __name__=='__main__':
    start_time=time.time()

    compute_date()

# id,
    end_time=time.time()
    print('耗时:',end_time-start_time)



