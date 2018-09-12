import pandas as pd
import numpy as np
import scipy as sc
import scipy.sparse as sp
from sklearn.utils import check_random_state 
from sklearn.model_selection import GridSearchCV,StratifiedKFold,train_test_split
from xgboost import XGBClassifier
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

from model_cv import modelfit_multi_cv
from feature_select import feature_selectfrommodel


#sys.path.append(utils.xgb_path)
import xgboost as xgb


import logging


from flags import FLAGS, unparsed



logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.DEBUG)

#gpu_dict={'gpu_id':0,'max_bin':16,'tree_method':['gpu_hist'],'predictor':['gpu_predictor']}
#gpu_dict={'gpu_id':0,'tree_method':'gpu_hist','updater':'grow_gpu'}
gpu_dict={'tree_method':'gpu_hist',}


#  gblinear  
#  gbtree 156

dart_param = {'booster': 'gbtree',
         'sample_type': 'uniform',
         'normalize_type': 'tree',
#         'rate_drop': 0.1,
#         'skip_drop': 0.5,
#         'num_round' : 50
         }

gbtree_param =dict(learning_rate =0.005,
        booster='gblinear',
        num_class=22,
        n_estimators=500,
        max_depth=3,
#        min_child_weight=5,
#        gamma=0.1,
#        subsample=0.7,
#        colsample_bytree=0.8,
#        return_train_score=True,
#        scoring='roc_auc',
#        scale_pos_weight=1,
#        reg_alpha=8,
#        reg_lambda=0.7,
#        rate_drop= 0.3,
#        skip_drop= 0.5,
        )

gbtree_param.update(dart_param)




def done(istrain='train'):
    columns=['hour_t1_start_hour_len_32', 'app_id_weight', 'every_app_len', 'type_no', 'times_len', 'brand', 'close_hour_len_t1_32', 'brand_cnt', 'app_len', 'app_time_t2_61', 'type_no_cnt', 'close_day_t1_32', 'hour_t1_close_hour_weight_32', 'close_hour_t1_32', 'hour_t1_start_hour_weight_32', 'hour_t2_start_hour_len_132', 'hour_t2_start_hour_len_94', 'close_hour_t1_33', 'close_day_t1_33', 'close_hour_t1_43', 'close_hour_t1_22', 'hour_t2_start_hour_len_124', 'hour_t2_start_hour_len_251', 'close_day_t1_19', 'close_hour_t1_36', 'hour_t1_start_hour_size_32', 'close_day_t1_4', 'close_day_size_t1_32', 'close_hour_t1_4', 'hour_t1_close_hour_weight_19', 'close_day_t1_36', 'close_day_t1_43', 'hour_t2_close_hour_weight_124', 'hour_t1_start_hour_len_33', 'close_hour_len_t1_33', 'hour_t2_start_hour_len_11', 'hour_t2_start_hour_len_158', 'close_hour_t1_19', 'hour_t2_start_hour_weight_132', 'close_day_size_t1_33', 'hour_t1_start_hour_size_36', 'hour_t1_start_hour_len_17', 'hour_t1_start_hour_len_43', 'app_time_t1_32',]
    
#    test_save.drop('click',axis=1,inplace=True)
#    op=['n_estimators','max_depth','min_child_weight','subsample','reg_alpha','gamma','fin']
    #  scale_pos_weight   rate_drop
    logging.debug(istrain) 
    op=['n_estimators']
    if istrain=='train':
        train_save = gdbt_data_get_train('n_class')
        
#        np.random.seed(999)
#        train_save = train_save.ix[r1 < 0.2, :]
        print(train_save.shape)
        y_train = train_save['n_class']
        train_save.drop('n_class',axis=1,inplace=True)
        X_train = train_save
#        X_train = train_save.ix[:,columns]
        
        
#        dtrain = xgb.DMatrix(X_train, label=y_train)
#        n_estimators = [i for i in range(200,1000,1)]
        xgb1 = XGBClassifier(**gbtree_param,
        objective='multi:softprob',
        eval_metric=['mlogloss',],
        nthread=-1,
        verbose=2,
        seed=27,
        silent=True,**gpu_dict)
        for i,oper in enumerate(op):
            modelfit_multi_cv(xgb1, X_train,y_train,cv_type=oper,)#random_state=i)        
            logging.debug(oper+":to save validation predictions ...")
            ret=dump(xgb1, FLAGS.tmp_data_path+'xgboost.cv_'+oper+'.model.joblib_dat') 
            logging.debug(ret)
            gc.collect()
        feature_selectfrommodel(xgb1, X_train,y_train)
        del train_save
        del X_train
        del y_train
    elif istrain=='eval':
        
        X_eval = gdbt_data_get_eval('n_class')
        print(X_eval.shape)
        y_eval = X_eval['n_class']
        X_eval.drop('n_class',axis=1,inplace=True)
        logging.debug(X_eval.shape)
#        X_eval = X_eval.ix[:,columns]
        for oper in op:
            xgb1 = load(FLAGS.tmp_data_path+'xgboost.cv_'+oper+'.model.joblib_dat')
            logging.debug(xgb1.get_params()['n_estimators'])
            dtrain_predprob = xgb1.predict_proba(X_eval)
            logging.debug(dtrain_predprob.shape)
            columns=[]
            for i in [1,2]:
                for j in range(11):
                    columns.append(str(i)+'-'+str(j))
            y_pred=pd.DataFrame(dtrain_predprob,columns=columns)
            def c(line):
                return [round(x,6) for x in line]
            y_pred.apply(lambda line:c(line),axis=1)
    
    
            logging.debug('-'*30)
            logging.debug(test_score(y_pred,y_eval))
        

        del X_eval
    elif istrain=='test':
        X_test = gdbt_data_get_test()
        print(X_test.shape)
#        X_test = X_test.ix[:,columns]
#        X_test.drop('click',axis=1,inplace=True)

        for oper in op:
            xgb1 = load(FLAGS.tmp_data_path+'xgboost.cv_'+oper+'.model.joblib_dat')
            logging.debug(xgb1.get_params()['n_estimators'])
            dtrain_predprob = xgb1.predict_proba(X_test)
            logging.debug(dtrain_predprob.shape)
            columns=[]
            for i in [1,2]:
                for j in range(11):
                    columns.append(str(i)+'-'+str(j))
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

            
            fin.to_csv(FLAGS.tmp_data_path+'1-'+oper+'-xgboost.test.csv',index=False)
        del X_test
        
        
if __name__ == "__main__":
    done(istrain='train')
    done(istrain='eval')
    done(istrain='test')
        

