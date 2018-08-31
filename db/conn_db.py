# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 20:12:49 2018

@author: admin
"""

import pymysql
import sys
sys.path.append('..')
from flags import FLAGS, unparsed

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
#sql="select * from documents"
#try:
#    cursor.execute(sql)
#    result=cursor.fetchall()
#    for i in result:
#        for j in i:
#            print(j,"=>",i[j])
#            print(" ")
#except:
#    print('error')
#db.close()