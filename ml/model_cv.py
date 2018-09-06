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
from sklearn.metrics import accuracy_score,f1_score


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
    
    X_train=X_train_part
    y_train=y_train_part
#    X_train = sp.csc_matrix(X_train)
#    X_val = sp.csc_matrix(X_val)
        
#    xgb_train = xgb.DMatrix(X_train_part, label=y_train_part)
#    xgb_test = xgb.DMatrix(X_val, label=y_val)


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
#        min_child_weight = range(1,6,2)
        param_cv = dict(max_depth=max_depth)

        cvresult = GridSearchCV(alg,param_grid=param_cv, scoring='neg_log_loss',n_jobs=8,pre_dispatch='n_jobs',cv=cv_folds,verbose=1)
        cvresult.fit(X_train,y_train)
        pd.DataFrame(cvresult.cv_results_).to_csv(FLAGS.tmp_data_path+'my_preds_maxdepth_min_child_weights_1.csv')
    #  
    
        print("Best: %f using %s" % (cvresult.best_score_, cvresult.best_params_))
        test_means = cvresult.cv_results_[ 'mean_test_score' ]
        test_stds = cvresult.cv_results_[ 'std_test_score' ]
        train_means = cvresult.cv_results_[ 'mean_train_score' ]
        train_stds = cvresult.cv_results_[ 'std_train_score' ]
                
        
        x_axis = range(0, max_depth)
        pyplot.errorbar(max_depth, test_means, yerr=test_stds ,label='Test')
        pyplot.errorbar(x_axis, train_means, yerr=train_stds ,label='Train')
        pyplot.title("XGBoost max_depth vs Log Loss")
        pyplot.xlabel( 'max_depth' )                                                                                                      
        pyplot.ylabel( 'Log Loss' )
        pyplot.savefig(FLAGS.tmp_data_path+'max_depth_1.png' )
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
        print("Best: %f using %s" % (cvresult.best_score_, cvresult.best_params_))
        test_means = cvresult.cv_results_[ 'mean_test_score' ]
        test_stds = cvresult.cv_results_[ 'std_test_score' ]
        train_means = cvresult.cv_results_[ 'mean_train_score' ]
        train_stds = cvresult.cv_results_[ 'std_train_score' ]
                
        
        x_axis = range(0, min_child_weight)
        pyplot.errorbar(max_depth, test_means, yerr=test_stds ,label='Test')
        pyplot.errorbar(x_axis, train_means, yerr=train_stds ,label='Train')
        pyplot.title("XGBoost min_child_weight vs Log Loss")
        pyplot.xlabel( 'min_child_weight' )                                                                                                      
        pyplot.ylabel( 'Log Loss' )
        pyplot.savefig(FLAGS.tmp_data_path+'min_child_weght_1.png' )

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
        
        print("Best: %f using %s" % (cvresult.best_score_, cvresult.best_params_))
        test_means = cvresult.cv_results_[ 'mean_test_score' ]
        test_stds = cvresult.cv_results_[ 'std_test_score' ]
        train_means = cvresult.cv_results_[ 'mean_train_score' ]
        train_stds = cvresult.cv_results_[ 'std_train_score' ]
        
        
        # plot results
        test_scores = np.array(test_means).reshape(len(subsample), len(colsample_bytree))
        train_scores = np.array(train_means).reshape(len(subsample), len(colsample_bytree))
        
        for i, value in enumerate(subsample):
            pyplot.plot(colsample_bytree, -test_scores[i], label= 'test_subsample:'   + str(value))
        for i, value in enumerate(colsample_bytree):
            pyplot.plot(subsample, train_scores[i], label= 'train_colsample_bytree:'   + str(value))

        pyplot.legend()
        pyplot.xlabel( 'subsample' )                                                                                                      
        pyplot.ylabel( 'Log Loss' )
        pyplot.savefig('subsample_vs_colsample_bytree_1.png' )
        
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
            
            
        print("Best: %f using %s" % (cvresult.best_score_, cvresult.best_params_))
        test_means = cvresult.cv_results_[ 'mean_test_score' ]
        test_stds = cvresult.cv_results_[ 'std_test_score' ]
        train_means = cvresult.cv_results_[ 'mean_train_score' ]
        train_stds = cvresult.cv_results_[ 'std_train_score' ]
        
        
        # plot results
        test_scores = np.array(test_means).reshape(len(reg_alpha), len(reg_lambda))
        train_scores = np.array(train_means).reshape(len(reg_alpha), len(reg_lambda))
        
        for i, value in enumerate(reg_alpha):
            pyplot.plot(reg_lambda, -test_scores[i], label= 'test_reg_alpha:'   + str(value))
        for i, value in enumerate(reg_lambda):
            pyplot.plot(reg_alpha, train_scores[i], label= 'train_reg_lambda:'   + str(value))

        pyplot.legend()
        pyplot.xlabel( 'reg_alpha' )                                                                                                      
        pyplot.ylabel( 'Log Loss' )
        pyplot.savefig('reg_alpha_vs_reg_lambda_1.png' )

    elif cv_type=='gamma':
        gamma = [i/10.0 for i in range(1,9)]    #default = 0, 测试0.1,1，1.5，2
        
        param_cv = dict(gamma=gamma)

        cvresult = GridSearchCV(alg,param_grid=param_cv, scoring='neg_log_loss',n_jobs=8,pre_dispatch='n_jobs',cv=cv_folds,verbose=1)
        cvresult.fit(X_train,y_train)
        pd.DataFrame(cvresult.cv_results_).to_csv(FLAGS.tmp_data_path+'gamma.csv')
    #  
        #最佳参数n_estimators
        logging.debug(cvresult.best_params_)
        for key,value in cvresult.best_params_.items():
            alg.set_params(**{key:value})
            
        #最佳参数n_estimators
        logging.debug(cvresult.best_params_)
        for key,value in cvresult.best_params_.items():
            alg.set_params(**{key:value})
        print("Best: %f using %s" % (cvresult.best_score_, cvresult.best_params_))
        test_means = cvresult.cv_results_[ 'mean_test_score' ]
        test_stds = cvresult.cv_results_[ 'std_test_score' ]
        train_means = cvresult.cv_results_[ 'mean_train_score' ]
        train_stds = cvresult.cv_results_[ 'std_train_score' ]
                
        
        x_axis = range(0, gamma)
        pyplot.errorbar(max_depth, test_means, yerr=test_stds ,label='Test')
        pyplot.errorbar(x_axis, train_means, yerr=train_stds ,label='Train')
        pyplot.title("XGBoost gamma vs Log Loss")
        pyplot.xlabel( 'gamma' )                                                                                                      
        pyplot.ylabel( 'Log Loss' )
        pyplot.savefig(FLAGS.tmp_data_path+'gamma.png' )

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
            
        #最佳参数n_estimators
        logging.debug(cvresult.best_params_)
        for key,value in cvresult.best_params_.items():
            alg.set_params(**{key:value})
        print("Best: %f using %s" % (cvresult.best_score_, cvresult.best_params_))
        test_means = cvresult.cv_results_[ 'mean_test_score' ]
        test_stds = cvresult.cv_results_[ 'std_test_score' ]
        train_means = cvresult.cv_results_[ 'mean_train_score' ]
        train_stds = cvresult.cv_results_[ 'std_train_score' ]
                
        
        x_axis = range(0, scale_pos_weight)
        pyplot.errorbar(max_depth, test_means, yerr=test_stds ,label='Test')
        pyplot.errorbar(x_axis, train_means, yerr=train_stds ,label='Train')
        pyplot.title("XGBoost scale_pos_weight vs Log Loss")
        pyplot.xlabel( 'scale_pos_weight' )                                                                                                      
        pyplot.ylabel( 'Log Loss' )
        pyplot.savefig(FLAGS.tmp_data_path+'scale_pos_weight.png' )
            
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
            
        print("Best: %f using %s" % (cvresult.best_score_, cvresult.best_params_))
        test_means = cvresult.cv_results_[ 'mean_test_score' ]
        test_stds = cvresult.cv_results_[ 'std_test_score' ]
        train_means = cvresult.cv_results_[ 'mean_train_score' ]
        train_stds = cvresult.cv_results_[ 'std_train_score' ]
        
        
        # plot results
        test_scores = np.array(test_means).reshape(len(rate_drop), len(skip_drop))
        train_scores = np.array(train_means).reshape(len(rate_drop), len(skip_drop))
        
        for i, value in enumerate(rate_drop):
            pyplot.plot(skip_drop, -test_scores[i], label= 'test_rate_drop:'   + str(value))
        for i, value in enumerate(skip_drop):
            pyplot.plot(rate_drop, train_scores[i], label= 'train_skip_drop:'   + str(value))

        pyplot.legend()
        pyplot.xlabel( 'rate_drop' )                                                                                                      
        pyplot.ylabel( 'Log Loss' )
        pyplot.savefig('reg_alpha_vs_reg_lambda_1.png' )
    #Fit the algorithm on the data
#    alg.set_params(cvresult.best_params_)
    alg.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)],eval_metric='mlogloss',)
        
    #Predict training set:
    
    train_predprob = alg.predict_proba(X_val)
    
    y_pred = alg.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    logging.debug('acc:'+str( acc*100.0)+'%')
#    f1 = f1_score(y_val, y_pred)
#    print(('f1_score', f1))
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