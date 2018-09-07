# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 17:41:47 2018

@author: admin
"""


# coding: utf-8

# # 聚类
# 
# 熟悉各中聚类算法的调用
# 并用评价指标选择合适的超参数

# In[1]:


#导入必要的工具包
import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.decomposition import PCA
import time
import sys
import logging

import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')

from collections import defaultdict
sys.path.append('..')
sys.path.append('../..')
from db.conn_db import db,cursor,engine,truncate_table,data_from_mysql,dev_id_train,get_package_label
from flags import FLAGS, unparsed
from data_preprocessing import *
# In[87]:


def time_profile(func):
    def warpper(*args,**kwargs):
        import time
        start=time.time()
        result=func(*args,**kwargs)
        end=time.time()
        print('花费时间为{}'.format(end-start))
        return result
    return warpper


# In[2]:

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.DEBUG)

file_path=FLAGS.file_path


# In[26]:


#读取训练数据
X_train =get_all_data_kmeans()


# In[79]:


X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min())
#X_train = (X_train ) / X_train.max()
X_train=X_train.fillna(0)
print(X_train.shape)
print(X_train.head(5))


#对数据进行PCA降维
pca = PCA(n_components=0.7)
pca.fit(X_train)

X_train_pca = pca.transform(X_train)

# 降维后的特征维数
print(X_train_pca.shape)


# In[105]:


# 将训练集合拆分成训练集和校验集，在校验集上找到最佳的模型超参数（PCA的维数）
X_train_part, X_val = train_test_split(X_train_pca,train_size = 0.8,random_state = 0)


# In[106]:


#拆分后的训练集和校验集的样本数目
print(X_train_part.shape)
print(X_val.shape)


# In[107]:


# 一个参数点（聚类数据为K）的模型，在校验集上评价聚类算法性能
@time_profile
def K_cluster_analysis(K, X_train, X_val):
    start = time.time()
    
    print("K-means begin with clusters: {}".format(K));
    
    #K-means,在训练集上训练
    mb_kmeans = MiniBatchKMeans(n_clusters = K)
    mb_kmeans.fit(X_train)
    
    # 在训练集和测试集上测试
    y_val_pred = mb_kmeans.predict(X_val)
    print(y_val_pred.shape)
    
    #以前两维特征打印训练数据的分类结果
    # K值的评估标准
    #常见的方法有轮廓系数Silhouette Coefficient和Calinski-Harabasz Index
    #这两个分数值越大则聚类效果越好
    #CH_score = metrics.calinski_harabaz_score(X_train,mb_kmeans.predict(X_train))
    CH_score = metrics.silhouette_score(X_train,mb_kmeans.predict(X_train))
    
    end = time.time()
    print("CH_score: {}, time elaps:{}".format(CH_score, int(end-start)))
    
    return CH_score


# In[108]:


# 设置超参数（聚类数目K）搜索范围
Ks = [2, 11, 22]
CH_scores = []

for K in Ks:
    ch = K_cluster_analysis(K, X_train_part, X_val)
    CH_scores.append(ch)


# In[109]:


print(np.array(CH_scores))


# In[111]:


# 绘制不同PCA维数下模型的性能，找到最佳模型／参数（分数最高）
#plt.plot(Ks, np.array(CH_scores), 'b-')


# 两个指标的变化趋势类似

# In[117]:


## 设置超参数（聚类数目K）搜索范围
#Ks = [x for x in range(2,11)]
#CH_scores2 = []
#
#for K in Ks:
#    ch = K_cluster_analysis(K, X_train_part, X_val)
#    CH_scores2.append(ch)


# In[116]:


# 绘制不同PCA维数下模型的性能，找到最佳模型／参数（分数最高）
#plt.plot(Ks, np.array(CH_scores2), 'b-')

