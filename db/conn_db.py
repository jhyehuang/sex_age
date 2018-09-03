# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 20:12:49 2018

@author: admin
"""

import pymysql
import sys
import pandas as pd
sys.path.append('..')
from flags import FLAGS

from sqlalchemy import create_engine

engine = create_engine('mysql+pymysql://'+FLAGS.user+':'+FLAGS.passwd+'@'+FLAGS.host+':3306/sex_age')


class mysql_conn(object):
    def __init__(self):
        self.db_conn=pymysql.connect(
            host=FLAGS.host,
            user=FLAGS.user,
            passwd=FLAGS.passwd,
            db=FLAGS.db,
            port=FLAGS.port,
            charset='utf8',
            cursorclass = pymysql.cursors.DictCursor
        )

    def __del__(self):
        self.db_conn.close()
db=mysql_conn().db_conn
cursor=db.cursor()

def truncate_table(table_name):
    sql='truncate table '+table_name
    return data_from_mysql(sql)

def data_from_mysql(sql):
    db.ping(reconnect=True)
    cursor.execute(sql)
    db.commit()
    ret=cursor.fetchall()
    ret=pd.DataFrame(line for line in ret)
    return ret

def dev_id_train():
    sql='select * from deviceid_train'
    deviceid_train=data_from_mysql(sql)
    return deviceid_train

def get_package_label(app_id,t):
    sql='select '+t+' from package_label where app_id=\"'
    sql=sql+app_id+'\"'
#    print (sql)
    ret=data_from_mysql(sql)
    return ret