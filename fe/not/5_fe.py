
# coding: utf-8

# In[49]:
import sys
sys.path.append('..')
import pandas as pd
from flags import FLAGS, unparsed
import numpy as np
from db.conn_db import db,cursor,engine,truncate_table,data_from_mysql,dev_id_train
import operator  
from functools import reduce
import time
from sklearn.preprocessing import LabelEncoder

from scipy.sparse import csr_matrix, hstack
import xgboost as xgb
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss
from data_preprocessing import *


# In[2]:


deviceid_packages=pd.read_csv(FLAGS.file_path+'deviceid_packages.csv',index_col='device_id')


# In[3]:


deviceid_package_start_close=pd.read_csv(FLAGS.file_path+'shichen_start_clost_times.csv')


# In[4]:


def def_hour_bin(hour):
    if hour>=18 and hour<=24:
        return 4
    elif hour>=0 and hour<8:
        return 0
    elif hour>=8 and hour<12:
        return 1
    elif hour>=12 and hour<14:
        return 2
    elif hour>=14 and hour<18:
        return 3


# In[5]:


def time_to_hour(timeStamp):
    # 字符类型的时间
    timeArray = time.localtime(int(timeStamp))
#    print(timeArray)
    # 转为时间数组
#    timeArray = time.strptime(timeArray, "%Y-%m-%d %H:%M:%S")
    return timeArray.tm_hour   # 2013


# In[6]:


deviceid_packages['row_no'] = np.arange(deviceid_packages.shape[0])


# In[7]:


# extract hour of the day from the timestamp 
deviceid_package_start_close['today_hour'] = deviceid_package_start_close['close'].map(lambda x:time_to_hour(x/1000))
deviceid_package_start_close['hour_bin'] = deviceid_package_start_close['today_hour'].apply(def_hour_bin)



# In[8]:


t1=deviceid_package_start_close.groupby(['device_id','hour_bin'])['hour_bin'].agg(['size'])


# In[9]:


t1.head(10)


# In[10]:


deviceid_package_start_close_hour_bin=(t1.merge(deviceid_packages[['row_no']], how='left',left_index=True, right_index=True).reset_index())


# In[11]:


ntod=5


# In[12]:


deviceid_package_start_close_hour_bin['size'] = deviceid_package_start_close_hour_bin['size'].map(lambda x: np.log(x+1))
t = deviceid_package_start_close_hour_bin.dropna(subset=['row_no'])


# In[13]:


Xtr_tod = csr_matrix((t['size'].values, (t.row_no, t.hour_bin)), 
                      shape=(deviceid_packages.shape[0],ntod))


# In[14]:


Xtr_tod.toarray()


# In[28]:


Xtrain = pd.DataFrame(data=Xtr_tod.toarray(),columns=['hour_0','hour_1','hour_2','hour_3','hour_4'])
Xtrain.shape


# In[29]:


Xtrain.head(5)


# In[36]:


deviceid_=pd.read_csv(FLAGS.file_path+'deviceid_packages.csv')


# In[41]:


deviceid_=pd.concat([deviceid_.ix[:,'device_id'],Xtrain],axis=1)
deviceid_.shape


# In[42]:


deviceid_.head(500)


# In[43]:


deviceid_.to_csv(FLAGS.file_path+'05_deviceid_packages.csv',index= False)

