import pandas as pd
import numpy as np
import scipy as sc
import scipy.sparse as sp
from sklearn.utils import check_random_state 
from sklearn.model_selection import train_test_split
import pylab 
import sys
import time
import utils
from utils import *
import os
from sklearn.ensemble import RandomForestClassifier
import logging


from joblib import dump, load
from flags import parse_args
FLAGS, unparsed = parse_args()

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.DEBUG)


train_save = load(FLAGS.tmp_data_path +FLAGS.train_job_name +'.joblib_dat')
x = train_save[FLAGS.train_col]
y = train_save[FLAGS.train_target]

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=4)

predv = 0
ctr = 0
nn = X_train.shape[0]

clf = RandomForestClassifier(n_estimators=32, max_depth=40, min_samples_split=100, min_samples_leaf=10, random_state=0, criterion='entropy',
                             max_features=8, verbose = 1, n_jobs=-1, bootstrap=False)



for i1 in range(8):
    clf.random_state = i1
    np.random.seed(i1)
    r1 = np.random.uniform(0, 1, nn)
    filter1 = np.logical_and(X_train, np.logical_and(r1 < .3, True))
    xt1 = X_train[filter1, :]
    yt1 = y_train[filter1]
    rf1 = clf.fit(xt1, yt1)
    y_hat = rf1.predict_proba(X_test)[:, 1]
    predv += y_hat
    ctr += 1
    ll = logloss(predv/ctr, y_test)
    logging.debug("iter", i1, ", logloss = ", ll)
    sys.stdout.flush()

feature_list = FLAGS.rf_feature_list

rf1_imp = pd.DataFrame({'feature':feature_list, 'impt': clf.feature_importances_})
logging.debug(rf1_imp.sort('impt'))

logging.debug("to save validation predictions ...")
dump(predv / ctr, FLAGS.tmp_data_path + 'rf_pred_v.joblib_dat')

