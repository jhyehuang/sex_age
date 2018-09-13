from os import chdir
import gc
import sys
import datetime
import pandas as pd
from flags import FLAGS, unparsed
import numpy as np
from db.conn_db import db,cursor,engine,truncate_table,data_from_mysql,dev_id_train
import operator  
from functools import reduce
import time

import logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.DEBUG)

####################################################### Load Data ####################################################### 



deviceid_package_start_close = pd.read_csv(FLAGS.file_path + "/09_01_deviceid_package_start_close.csv")
deviceid_package_start_close = deviceid_package_start_close.drop_duplicates()

####################################################### Functions ####################################################### 

def extract_hour(datetime):
    datetime = datetime[11:13]
    return(int(datetime))
def time_to_hour(timeStamp):
    # 字符类型的时间
    timeArray = time.localtime(int(timeStamp))
#    print(timeArray)
    # 转为时间数组
#    timeArray = time.strptime(timeArray, "%Y-%m-%d %H:%M:%S")
    return timeArray.tm_hour   # 2013
def set_day_time(hour):
    if (hour < 6):
        return("dawn")
    if (6 <= hour & hour < 8):
        return("breakfast")
    if (8 <= hour & hour < 12):
        return("morning")
    if (12 <= hour & hour < 14):
        return("lunch")
    if (14 <= hour & hour < 18):
        return("afternoon")
    if (18 <= hour & hour < 20):
        return("dinner")
    else: # (20 <= hour && hour < 0) {
        return("night")

def time_to_week(timeStamp):
    timeArray =int(time.mktime(time.strptime(timeStamp,"%Y%m%d")))
    date = datetime.datetime.fromtimestamp(timeArray)
    day=date.weekday()
#    logging.debug(day)
    return day   # 2013

def set_weekday(timestamp):
    WEEKDAY = {0: "monday", 1: "tuesday", 2: "wednesday", 3: "thursday", 4: "friday", 5: "saturday", 6: "sunday"}
    n_weekday = time_to_week(timestamp/1000)
    return WEEKDAY[n_weekday]

####################################################### Execution ####################################################### 

deviceid_package_start_close['today_hour'] = deviceid_package_start_close['close'].map(lambda x:time_to_hour(x/1000))

deviceid_package_start_close = deviceid_package_start_close.sort_values(['today_hour'], ascending=True)

deviceid_package_start_close["daytime"] = deviceid_package_start_close["today_hour"].apply(set_day_time)

deviceid_package_start_close = deviceid_package_start_close.sort_values(['event_id'], ascending=True)

deviceid_package_start_close["weekday"] = pd.to_datetime(deviceid_package_start_close["close"]).apply(set_weekday)
####################################################### Save Data ####################################################### 

deviceid_package_start_close.to_csv(FLAGS.file_path + "events_v2.csv", sep = ";", index=False)

