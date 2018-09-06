# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 11:25:02 2018

@author: admin
"""
import pandas as pd
import numpy as np
import scipy as sc
import scipy.sparse as sp
from sklearn.utils import check_random_state 
from sklearn.model_selection import GridSearchCV,StratifiedKFold,train_test_split

from sklearn.metrics import log_loss
import xgboost as xgb

from matplotlib import pyplot
pyplot.switch_backend('agg')
import pylab 
import sys
sys.path.append('..')
import time
from joblib import dump, load, Parallel, delayed
import utils
import gc
from data_preprocessing import *


import logging


from flags import FLAGS, unparsed


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.DEBUG)

#gpu_dict={'gpu_id':0,'max_bin':16,'tree_method':['gpu_hist'],'predictor':['gpu_predictor']}
#gpu_dict={'gpu_id':0,'tree_method':'gpu_hist','updater':'grow_gpu'}
gpu_dict={'tree_method':'gpu_hist',}

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)


def modelfit_binary_cv(alg, X_train, y_train,cv_folds=kfold, early_stopping_rounds=10,cv_type='n_estimators',random_state=0):
    X_train_part, X_val, y_train_part, y_val = train_test_split(X_train, y_train, train_size = 0.6,random_state = random_state)
    if cv_type=='n_estimators':
        xgb_param = alg.get_xgb_params()
#        xgb_param['num_class'] = 2

        xgtrain = xgb.DMatrix(X_train, label = y_train)
        
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], folds =cv_folds,
                         metrics='auc', early_stopping_rounds=early_stopping_rounds)
        
        n_estimators = cvresult.shape[0]
        alg.set_params(n_estimators = n_estimators)
#        for key,value in cvresult.best_params_.items():
#            alg.set_params(**{key:value})
        print (cvresult)

        cvresult.to_csv( FLAGS.tmp_data_path+'n_estimators.csv', index_label = 'n_estimators')
        
        # plot
#        test_means = cvresult['test-logloss-mean']
#        test_stds = cvresult['test-logloss-std'] 
        
#        train_means = cvresult['train-logloss-mean']
#        train_stds = cvresult['train-logloss-std'] 

#        x_axis = range(0, n_estimators)
#        pyplot.errorbar(x_axis, test_means, yerr=test_stds ,label='Test')
#        pyplot.errorbar(x_axis, train_means, yerr=train_stds ,label='Train')
#        pyplot.title("XGBoost n_estimators vs Log Loss")
#        pyplot.xlabel( 'n_estimators' )
#        pyplot.ylabel( 'Log Loss' )
#        pyplot.savefig(  FLAGS.tmp_data_path+'n_estimators4_2_3_699.png' )
        
    elif cv_type=='max_depth':
#        xgb_param = alg.get_xgb_params()
        max_depth = range(4,7,1)
#        min_child_weight = range(1,6,1)
        param_cv = dict(max_depth=max_depth)

        cvresult = GridSearchCV(alg,param_grid=param_cv, scoring='neg_log_loss',n_jobs=8,pre_dispatch='n_jobs',cv=cv_folds,verbose=2)
        cvresult.fit(X_train,y_train)
        pd.DataFrame(cvresult.cv_results_).to_csv(FLAGS.tmp_data_path+'my_preds_maxdepth_min_child_weights_1.csv')
    #  
        #最佳参数n_estimators
        logging.debug(cvresult.best_params_)
        for key,value in cvresult.best_params_.items():
            alg.set_params(**{key:value})
    elif cv_type=='min_child_weight':
#        xgb_param = alg.get_xgb_params()
        min_child_weight = range(1,6,1)
        param_cv = dict(min_child_weight=min_child_weight)

        cvresult = GridSearchCV(alg,param_grid=param_cv, scoring='neg_log_loss',n_jobs=8,pre_dispatch='n_jobs',cv=cv_folds,verbose=2)
        cvresult.fit(X_train,y_train)
        pd.DataFrame(cvresult.cv_results_).to_csv(FLAGS.tmp_data_path+'my_preds_maxdepth_min_child_weights_1.csv')
    #  
        #最佳参数n_estimators
        logging.debug(cvresult.best_params_)
        for key,value in cvresult.best_params_.items():
            alg.set_params(**{key:value})
        

    elif cv_type=='subsample':
        subsample = [i/10.0 for i in range(6,9)]
        colsample_bytree = [i/10.0 for i in range(6,10)]
        param_cv = dict(subsample=subsample, colsample_bytree=colsample_bytree)

        cvresult = GridSearchCV(alg,param_grid=param_cv, scoring='neg_log_loss',n_jobs=8,pre_dispatch='n_jobs',cv=cv_folds,verbose=2)
        cvresult.fit(X_train,y_train)
        pd.DataFrame(cvresult.cv_results_).to_csv(FLAGS.tmp_data_path+'my_preds_subsampleh_colsample_bytree_1.csv')
    #  
        #最佳参数n_estimators
        logging.debug(cvresult.best_params_)
        for key,value in cvresult.best_params_.items():
            alg.set_params(**{key:value})
        
    elif cv_type=='reg_alpha':
        reg_alpha = [i/10.0 for i in range(60,80)]    #default = 0, 测试0.1,1，1.5，2
        reg_lambda =[i/10 for i in range(4,9)]      #default = 1，测试0.1， 0.5， 1，2
        
        param_cv = dict(reg_alpha=reg_alpha, reg_lambda=reg_lambda)

        cvresult = GridSearchCV(alg,param_grid=param_cv, scoring='neg_log_loss',n_jobs=8,pre_dispatch='n_jobs',cv=cv_folds,verbose=2)
        cvresult.fit(X_train,y_train)
        pd.DataFrame(cvresult.cv_results_).to_csv(FLAGS.tmp_data_path+'reg_alpha_vs_reg_lambda1.csv')
    #  
        #最佳参数n_estimators
        logging.debug(cvresult.best_params_)
        for key,value in cvresult.best_params_.items():
            alg.set_params(**{key:value})

    elif cv_type=='gamma':
        gamma = [i/10.0 for i in range(1,9)]    #default = 0, 测试0.1,1，1.5，2
        
        param_cv = dict(gamma=gamma)

        cvresult = GridSearchCV(alg,param_grid=param_cv, scoring='neg_log_loss',n_jobs=8,pre_dispatch='n_jobs',cv=cv_folds,verbose=2)
        cvresult.fit(X_train,y_train)
        pd.DataFrame(cvresult.cv_results_).to_csv(FLAGS.tmp_data_path+'gamma.csv')
    #  
        #最佳参数n_estimators
        logging.debug(cvresult.best_params_)
        for key,value in cvresult.best_params_.items():
            alg.set_params(**{key:value})

    elif cv_type=='scale_pos_weight':
        scale_pos_weight = [i for i in range(1,5)]    #default = 0, 测试0.1,1，1.5，2
        
        param_cv = dict(scale_pos_weight=scale_pos_weight)

        cvresult = GridSearchCV(alg,param_grid=param_cv, scoring='neg_log_loss',n_jobs=8,pre_dispatch='n_jobs',cv=cv_folds,verbose=2)
        cvresult.fit(X_train,y_train)
        pd.DataFrame(cvresult.cv_results_).to_csv(FLAGS.tmp_data_path+'scale_pos_weight.csv')
    #  
        #最佳参数n_estimators
        logging.debug(cvresult.best_params_)
        for key,value in cvresult.best_params_.items():
            alg.set_params(**{key:value})
            
    elif cv_type=='rate_drop':
        rate_drop = [i/10 for i in range(1,7)]    #default = 0, 测试0.1,1，1.5，2
        skip_drop = [i/10 for i in range(1,7)]
        
        param_cv = dict(rate_drop=rate_drop,skip_drop=skip_drop)

        cvresult = GridSearchCV(alg,param_grid=param_cv, scoring='neg_log_loss',n_jobs=8,pre_dispatch='n_jobs',cv=cv_folds,verbose=2)
        cvresult.fit(X_train,y_train)
        pd.DataFrame(cvresult.cv_results_).to_csv(FLAGS.tmp_data_path+'rate_drop.csv')
    #  
        #最佳参数n_estimators
        logging.debug(cvresult.best_params_)
        for key,value in cvresult.best_params_.items():
            alg.set_params(**{key:value})
    #Fit the algorithm on the data
#    alg.set_params(cvresult.best_params_)
    alg.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)],eval_metric='auc',)
        
    #Predict training set:
    
    train_predprob = alg.predict_proba(X_val)
    try:
        logloss = log_loss(y_val, train_predprob)
        logging.debug(logloss)
    except:
        pass
    
    try:
        _,lloss = logloss(train_predprob[:,1],y_val)

       #Print model report:
        logging.debug ("logloss of train :" )
        logging.debug(lloss)
    except:
        pass

def modelfit_multi_cv(alg, X_train, y_train,cv_folds=kfold, early_stopping_rounds=10,cv_type='n_estimators',random_state=173):
    X_train_part, X_val, y_train_part, y_val = train_test_split(X_train, y_train, train_size = 0.8,random_state = random_state)
    if cv_type=='n_estimators':
        xgb_param = alg.get_xgb_params()
        logging.debug(xgb_param['num_class'])
        

        xgtrain = xgb.DMatrix(X_train, label = y_train)
        
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], folds =cv_folds,
                         metrics='mlogloss', early_stopping_rounds=early_stopping_rounds)
        
        n_estimators = cvresult.shape[0]
        alg.set_params(n_estimators = n_estimators)
#        for key,value in cvresult.best_params_.items():
#            alg.set_params(**{key:value})
        print (cvresult)

        cvresult.to_csv( FLAGS.tmp_data_path+'n_estimators.csv', index_label = 'n_estimators')
        
        # plot
        test_means = cvresult['test-mlogloss-mean']
        test_stds = cvresult['test-mlogloss-std'] 
        
        train_means = cvresult['train-mlogloss-mean']
        train_stds = cvresult['train-mlogloss-std'] 

        x_axis = range(0, n_estimators)
        pyplot.errorbar(x_axis, test_means, yerr=test_stds ,label='Test')
        pyplot.errorbar(x_axis, train_means, yerr=train_stds ,label='Train')
        pyplot.title("XGBoost n_estimators vs Log Loss")
        pyplot.xlabel( 'n_estimators' )
        pyplot.ylabel( 'Log Loss' )
        pyplot.savefig(  FLAGS.tmp_data_path+'n_estimators4_2_3_699.png' )
        
    elif cv_type=='max_depth':
#        xgb_param = alg.get_xgb_params()
        max_depth = range(4,7,1)
#        min_child_weight = range(1,6,1)
        param_cv = dict(max_depth=max_depth)

        cvresult = GridSearchCV(alg,param_grid=param_cv, scoring='neg_log_loss',n_jobs=8,pre_dispatch='n_jobs',cv=cv_folds,verbose=1)
        cvresult.fit(X_train,y_train)
        pd.DataFrame(cvresult.cv_results_).to_csv(FLAGS.tmp_data_path+'my_preds_maxdepth_min_child_weights_1.csv')
    #  
        #最佳参数n_estimators
        logging.debug(cvresult.best_params_)
        for key,value in cvresult.best_params_.items():
            alg.set_params(**{key:value})
    elif cv_type=='min_child_weight':
#        xgb_param = alg.get_xgb_params()
        min_child_weight = range(1,6,1)
        param_cv = dict(min_child_weight=min_child_weight)

        cvresult = GridSearchCV(alg,param_grid=param_cv, scoring='neg_log_loss',n_jobs=8,pre_dispatch='n_jobs',cv=cv_folds,verbose=1)
        cvresult.fit(X_train,y_train)
        pd.DataFrame(cvresult.cv_results_).to_csv(FLAGS.tmp_data_path+'my_preds_maxdepth_min_child_weights_1.csv')
    #  
        #最佳参数n_estimators
        logging.debug(cvresult.best_params_)
        for key,value in cvresult.best_params_.items():
            alg.set_params(**{key:value})
        

    elif cv_type=='subsample':
        subsample = [i/10.0 for i in range(6,9)]
        colsample_bytree = [i/10.0 for i in range(6,10)]
        param_cv = dict(subsample=subsample, colsample_bytree=colsample_bytree)

        cvresult = GridSearchCV(alg,param_grid=param_cv, scoring='neg_log_loss',n_jobs=8,pre_dispatch='n_jobs',cv=cv_folds,verbose=1)
        cvresult.fit(X_train,y_train)
        pd.DataFrame(cvresult.cv_results_).to_csv(FLAGS.tmp_data_path+'my_preds_subsampleh_colsample_bytree_1.csv')
    #  
        #最佳参数n_estimators
        logging.debug(cvresult.best_params_)
        for key,value in cvresult.best_params_.items():
            alg.set_params(**{key:value})
        
    elif cv_type=='reg_alpha':
        reg_alpha = [i/10.0 for i in range(60,80)]    #default = 0, 测试0.1,1，1.5，2
        reg_lambda =[i/10 for i in range(4,9)]      #default = 1，测试0.1， 0.5， 1，2
        
        param_cv = dict(reg_alpha=reg_alpha, reg_lambda=reg_lambda)

        cvresult = GridSearchCV(alg,param_grid=param_cv, scoring='neg_log_loss',n_jobs=8,pre_dispatch='n_jobs',cv=cv_folds,verbose=1)
        cvresult.fit(X_train,y_train)
        pd.DataFrame(cvresult.cv_results_).to_csv(FLAGS.tmp_data_path+'reg_alpha_vs_reg_lambda1.csv')
    #  
        #最佳参数n_estimators
        logging.debug(cvresult.best_params_)
        for key,value in cvresult.best_params_.items():
            alg.set_params(**{key:value})

    elif cv_type=='gamma':
        gamma = [i/10.0 for i in range(1,9)]    #default = 0, 测试0.1,1，1.5，2
        
        param_cv = dict(gamma=gamma)

        cvresult = GridSearchCV(alg,param_grid=param_cv, scoring='neg_log_loss',n_jobs=8,pre_dispatch='n_jobs',cv=cv_folds,verbose=1)
        cvresult.fit(X_train,y_train)
        pd.DataFrame(cvresult.cv_results_).to_csv('gamma.csv')
    #  
        #最佳参数n_estimators
        logging.debug(cvresult.best_params_)
        for key,value in cvresult.best_params_.items():
            alg.set_params(**{key:value})

    elif cv_type=='scale_pos_weight':
        scale_pos_weight = [i for i in range(1,5)]    #default = 0, 测试0.1,1，1.5，2
        
        param_cv = dict(scale_pos_weight=scale_pos_weight)

        cvresult = GridSearchCV(alg,param_grid=param_cv, scoring='neg_log_loss',n_jobs=8,pre_dispatch='n_jobs',cv=cv_folds,verbose=1)
        cvresult.fit(X_train,y_train)
        pd.DataFrame(cvresult.cv_results_).to_csv(FLAGS.tmp_data_path+'scale_pos_weight.csv')
    #  
        #最佳参数n_estimators
        logging.debug(cvresult.best_params_)
        for key,value in cvresult.best_params_.items():
            alg.set_params(**{key:value})
            
    elif cv_type=='rate_drop':
        rate_drop = [i/10 for i in range(1,7)]    #default = 0, 测试0.1,1，1.5，2
        skip_drop = [i/10 for i in range(1,7)]
        
        param_cv = dict(rate_drop=rate_drop,skip_drop=skip_drop)

        cvresult = GridSearchCV(alg,param_grid=param_cv, scoring='neg_log_loss',n_jobs=8,pre_dispatch='n_jobs',cv=cv_folds,verbose=1)
        cvresult.fit(X_train,y_train)
        pd.DataFrame(cvresult.cv_results_).to_csv(FLAGS.tmp_data_path+'rate_drop.csv')
    #  
        #最佳参数n_estimators
        logging.debug(cvresult.best_params_)
        for key,value in cvresult.best_params_.items():
            alg.set_params(**{key:value})
    #Fit the algorithm on the data
#    alg.set_params(cvresult.best_params_)
    alg.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)],eval_metric=['auc'],)
        
    #Predict training set:
    
    train_predprob = alg.predict_proba(X_val)
    try:
        logloss = log_loss(y_val, train_predprob)
        logging.debug(logloss)
    except:
        pass
    
    try:
        _,lloss = logloss(train_predprob[:,1],y_val)

       #Print model report:
        logging.debug ("logloss of train :" )
        logging.debug(lloss)
    except:
        pass


if __name__ == "__main__":
    pass