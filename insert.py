from db.conn_db import db,cursor,engine,truncate_table,data_from_mysql
from flags import FLAGS, unparsed
import pandas as pd
data_src=pd.read_csv(FLAGS.file_path+'09_01_deviceid_package_start_close.csv')
table_name='deviceid_package_start_close'
truncate_table(table_name)
print(table_name)
print(data_src.head(5))
pd.io.sql.to_sql(data_src,table_name, engine,if_exists='append', index= False)
