# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 15:53:52 2018

@author: admin
"""

from numpy import sort
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from data_preprocessing import *
from sklearn.metrics import accuracy_score
import pandas as  pd
import sys
sys.path.append('..')
import time
from joblib import dump, load, Parallel, delayed
import utils
from data_preprocessing import *
import logging


from flags import FLAGS, unparsed



logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.DEBUG)

gpu_dict={'tree_method':'gpu_hist',}

def feature_selectfrommodel(model_XGB,X_train,y_train):
    # Fit model using each importance as a threshold
    X_train_part, X_test, y_train_part, y_test = train_test_split(X_train, y_train, train_size = 0.8,random_state = 173)
    thresholds = sort(model_XGB.feature_importances_)
    
    ### 特征选择
    df = pd.DataFrame(X_train.columns.tolist(), columns=['feature'])
    df['importance']=list(model_XGB.feature_importances_)                           # 特征分数
    df = df.sort_values(by='importance',ascending=False)                      # 特征排序
    df1=df.ix[df.importance.values>0.0,:]
    logging.debug(df1['feature'].tolist())
    dump(df1['feature'].tolist(), FLAGS.tmp_data_path+'feature.joblib_dat') 
    df1.to_csv(FLAGS.tmp_data_path+'feature_score.csv',index=None,encoding='utf-8') # 保存分数
    logging.debug(df) 
#    for thresh in thresholds:
#      # select features using threshold
#      xgb_param = model_XGB.get_xgb_params()
#      xgb_param.update(gpu_dict)
#      selection = SelectFromModel(model_XGB, threshold=thresh, prefit=True)
#      select_X_train = selection.transform(X_train)
#      # train model
#      selection_model = XGBClassifier(**xgb_param)
#      selection_model.fit(select_X_train, y_train)
#    # eval model
#      select_X_test = selection.transform(X_test)
#      y_pred = selection_model.predict(select_X_test)
#      predictions = [round(value) for value in y_pred]
#      accuracy = accuracy_score(y_test, predictions)
#      print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1],
#          accuracy*100.0))