import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.metrics import log_loss

from matplotlib import pyplot
pyplot.switch_backend('agg')
import sys
sys.path.append('..')
from joblib import dump, load, Parallel, delayed
import gc
from data_preprocessing import *
from model_cv import modelfit_multi_cv
#sys.path.append(utils.xgb_path)
import logging

from flags import FLAGS, unparsed
from sklearn.metrics import accuracy_score,f1_score



logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.DEBUG)

#gpu_dict={'gpu_id':0,'max_bin':16,'tree_method':['gpu_hist'],'predictor':['gpu_predictor']}
#gpu_dict={'gpu_id':0,'tree_method':'gpu_hist','updater':'grow_gpu'}
gpu_dict={'tree_method':'gpu_hist',}

gbtree_param =dict(learning_rate =0.1,
    booster='gbtree',
    num_class=11,
    n_estimators=180,
#        n_estimators=1219,
#        n_estimators=1,
    max_depth=3,
#        min_child_weight=5,
#        gamma=0.1,
#        subsample=0.6,
#        colsample_bytree=0.9,
#        return_train_score=True,
#    scoring='roc_auc',
#        scale_pos_weight=1,
#    reg_alpha=8,
#    reg_lambda=0.7,
#    rate_drop= 0.3,
#    skip_drop= 0.5,
    )


def done(istrain,X_train,y_train,flag):
#    test_save.drop('click',axis=1,inplace=True)
#    op=['n_estimators','max_depth','min_child_weight','subsample','reg_alpha','gamma','fin']
    op=['max_depth']

    if istrain=='train':
        xgb1 = XGBClassifier(**gbtree_param,
        objective='multi:softprob',
        eval_metric=['mlogloss'],
        nthread=-1,
        verbose=1,
        seed=27,
        silent=True,**gpu_dict)
        for i,oper in enumerate(op):
            modelfit_multi_cv(xgb1, X_train,y_train,cv_type=oper,random_state=i)        
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
#            dtrain_predprob = xgb1.predict_proba(X_train)
            y_pred = xgb1.predict(X_train)
        
            acc = accuracy_score(y_train, y_pred)
            logging.debug('acc:'+str( acc*100.0)+'%')
    
    
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
            df1=pd.read_csv(FLAGS.tmp_data_path+'sex_fin-xgboost.test.csv')
            test_concat(df1,fin)
        del X_train
        return fin
        
        
def headle_age(flag):
    train_save = gdbt_data_get_train(flag)
    print(train_save.shape)
    train_save[flag]=train_save[flag].astype('category').values.codes
    y_train = train_save[flag]
    train_save.drop(flag,axis=1,inplace=True)

    
    logging.debug(train_save.shape)
    logging.debug(y_train.unique())
    done('train',train_save,y_train,flag)
    
    X_eval = gdbt_data_get_eval(flag)
    print(X_eval.shape)
    y_eval = X_eval[flag]
    logging.debug(X_eval.shape)
    X_eval.drop(flag,axis=1,inplace=True)
    done('eval',X_eval,y_eval,flag)
    
    X_test = gdbt_data_get_test(flag)
    print(X_test.shape)
    y=None

    done('test',X_test,y,flag)

        
def test_concat(df1,df2):
    columns=[]
    columns.append('DeviceID')
    for i in [1,2]:
        for j in range(11):
            col=str(i)+'-'+str(j)
            columns.append(col)
            df1[col]=df1[str(i)]*df2[str(j)]
    def c(line):
        return [round(x,6) for x in line[1:]]
    df1.apply(lambda line:c(line),axis=1)
    df1.to_csv(FLAGS.tmp_data_path+'2x11_xgboost.test.csv',columns=columns,index=False)
if __name__ == "__main__":
    headle_age('age')

    
