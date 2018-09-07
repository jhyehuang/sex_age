# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 15:53:52 2018

@author: admin
"""

from numpy import sort
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier
from data_preprocessing import *
from sklearn.metrics import accuracy_score

def feature_selectfrommodel(model_XGB,X_train,y_train,X_test,y_test):
    # Fit model using each importance as a threshold
    thresholds = sort(model_XGB.feature_importances_)
    for thresh in thresholds:
      # select features using threshold
      selection = SelectFromModel(model_XGB, threshold=thresh, prefit=True)
      select_X_train = selection.transform(X_train)
      # train model
      selection_model = XGBClassifier()
      selection_model.fit(select_X_train, y_train)
    # eval model
      select_X_test = selection.transform(X_test)
      y_pred = selection_model.predict(select_X_test)
      predictions = [round(value) for value in y_pred]
      accuracy = accuracy_score(y_test, predictions)
      print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1],
          accuracy*100.0))