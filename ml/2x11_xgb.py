import pandas as pd
import numpy as np
import scipy as sc
import scipy.sparse as sp
from sklearn.utils import check_random_state 
from sklearn.model_selection import GridSearchCV,StratifiedKFold,train_test_split
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.metrics import log_loss

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



#sys.path.append(utils.xgb_path)
import xgboost as xgb


import logging


from flags import FLAGS, unparsed


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.DEBUG)

#gpu_dict={'gpu_id':0,'max_bin':16,'tree_method':['gpu_hist'],'predictor':['gpu_predictor']}
#gpu_dict={'gpu_id':0,'tree_method':'gpu_hist','updater':'grow_gpu'}
gpu_dict={'tree_method':'gpu_hist',}

#  valid values are: {'approx', 'auto', 'exact', 'gpu_exact', 'gpu_hist', 'hist'}"
#params=param.update(gpu_dict)
#直接调用xgboost内嵌的交叉验证（cv），可对连续的n_estimators参数进行快速交叉验证
#而GridSearchCV只能对有限个参数进行交叉验证



def modelfit_cv(alg, X_train, y_train,cv_folds=None, early_stopping_rounds=10,cv_type='n_estimators',random_state=173):
    X_train_part, X_val, y_train_part, y_val = train_test_split(X_train, y_train, train_size = 0.8,random_state = random_state)
    if cv_type=='n_estimators':
        xgb_param = alg.get_xgb_params()
#        xgb_param['num_class'] = 2

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
        pd.DataFrame(cvresult.cv_results_).to_csv('gamma.csv')
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
    alg.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)],eval_metric='mlogloss',)
        
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


gbtree_param =dict(learning_rate =0.1,
    booster='gbtree',
    num_class=None,
    n_estimators=75,
#        n_estimators=1219,
#        n_estimators=1,
    max_depth=6,
#        min_child_weight=5,
#        gamma=0.1,
#        subsample=0.7,
#        colsample_bytree=0.8,
#        return_train_score=True,
#    scoring='roc_auc',
#        scale_pos_weight=1,
#    reg_alpha=8,
#    reg_lambda=0.7,
#    rate_drop= 0.3,
#    skip_drop= 0.5,
    )

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)

def done(istrain,X_train,y_train,flag):
#    test_save.drop('click',axis=1,inplace=True)
#    op=['n_estimators','max_depth','min_child_weight','subsample','reg_alpha','gamma','fin']
    op=['fin']
    gbtree_param.update(dict(num_class=len(y_train.unique().tolist())))
    if istrain=='train':
        xgb1 = XGBClassifier(**gbtree_param,
        objective='multi:softprob',
        eval_metric=['mlogloss'],
        nthread=-1,
        verbose=2,
        seed=27,
        silent=True,**gpu_dict)
        for i,oper in enumerate(op):
            modelfit_cv(xgb1, X_train,y_train, cv_folds = kfold,cv_type=oper,random_state=i)        
            logging.debug(oper+":to save validation predictions ...")
            ret=dump(xgb1, FLAGS.tmp_data_path+flag+'_xgboost.cv_'+oper+'.model.joblib_dat') 
            logging.debug(ret)
            gc.collect()
        del X_train
        del y_train
    elif istrain=='eval':
        for oper in op:
            xgb1 = load(FLAGS.tmp_data_path+flag+'_xgboost.cv_'+oper+'.model.joblib_dat')
            logging.debug(xgb1.get_params()['n_estimators'])
            dtrain_predprob = xgb1.predict_proba(X_train)
            logging.debug(dtrain_predprob.shape)
            columns=[]
            for i in range(dtrain_predprob.shape[1]):
                if flag=='sex':
                    columns.append(str(i+1))
                else:
                    columns.append(str(i))
            y_pred=pd.DataFrame(dtrain_predprob,columns=columns)
            def c(line):
                return [round(x,6) for x in line]
            y_pred.apply(lambda line:c(line),axis=1)
    
    
            logging.debug('-'*30)
            
        

        del X_train
    elif istrain=='test':

        for oper in op:
            xgb1 = load(FLAGS.tmp_data_path+flag+'_xgboost.cv_'+oper+'.model.joblib_dat')
            logging.debug(xgb1.get_params()['n_estimators'])
            dtrain_predprob = xgb1.predict_proba(X_train)
            logging.debug(dtrain_predprob.shape)
            columns=[]
            for i in range(dtrain_predprob.shape[1]):
                if flag=='sex':
                    columns.append(str(i+1))
                else:
                    columns.append(str(i))
            y_pred=pd.DataFrame(dtrain_predprob,columns=columns)
            def c(line):
                return [round(x,6) for x in line]
            y_pred.apply(lambda line:c(line),axis=1)


            logging.debug('-'*30)
#            y_pred=np.array(y_pred).reshape(-1,1)
            logging.debug(y_pred)
            test_id=pd.read_csv(FLAGS.file_path+'deviceid_test.csv')
            logging.debug(test_id['device_id'].shape)
            test_id['device_id']=test_id['device_id'].map(str)
            test_id.rename(columns={'device_id':'DeviceID'}, inplace = True)
            fin=pd.concat([test_id,y_pred],axis=1)
            
            print(fin)

            
            fin.to_csv(FLAGS.tmp_data_path+flag+'_'+oper+'-xgboost.test.csv',index=False)
        del X_train
        return fin
        
        
def headle_sex(flag):
    train_save = gdbt_data_get_train()
    print(train_save.shape)
    y_train = train_save[flag]
    train_save.drop(flag,axis=1,inplace=True)
    done('train',train_save,y_train,flag)
    
    X_eval = gdbt_data_get_eval()
    print(X_eval.shape)
    y_eval = X_eval[flag]
    X_eval.drop(flag,axis=1,inplace=True)
    logging.debug(X_eval.shape)
    
    done('eval',X_eval,y_eval,flag)
    
    X_test = gdbt_data_get_test()
    print(X_test.shape)
    y=None

    done('test',X_test,y,flag)
        
def test_concat(df1,df2):
    for i in [1,2]:
        for j in range(11):
            col=str(i)+'-'+str(j)
            columns.append(col)
            df1[col]=df1[str(i)]*df2[str(j)]
    columns.append('DeviceID')
    df1.to_csv(FLAGS.tmp_data_path+'2x11_xgboost.test.csv',columns=columns,index=False)
if __name__ == "__main__":
    result=[]
    for col in ['sex','age']:
        result.append(headle_sex(col))
        
    test_concat(result[0],result[1])
    
