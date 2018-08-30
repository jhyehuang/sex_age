# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 20:12:49 2018

@author: admin
"""

import pymysql

class mysql_conn(object):
    def __init__(self):
        host='localhost'
        user='root'
        passwd='root'
        port=3306
        db='sex_age'
        self.db_conn=pymysql.connect(
            host=host,
            user=user,
            passwd=passwd,
            db=db,
            port=port,
            charset='utf8',
            cursorclass = pymysql.cursors.DictCursor
        )
        return self.db_conn
    def __del__(self):
        self.db_conn.close()
db=mysql_conn()
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