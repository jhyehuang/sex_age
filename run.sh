#!/bin/bash

starttime=$(date +%s)


cd data_writer/
python queue_to_csv.py

cd ../arg_data_pre/
python arg_data_pre.py

cd ../data_anly/
sh kill_worker.sh
nohup python data_anly_worker.py --flag=tel --get_address=True &

cd ../es_crawler/
pwd
python es_crawler.py --flag=tel --match_type=term --get_address=True
echo 'tel term'

#python es_crawler.py --flag=name --match_type=term --term_flag=True
#sleep 5
echo 'name term'


cd ../data_writer/
python queue_to_csv.py --match_type=term --get_address=True

echo 'match'
cd ../es_crawler/
python es_crawler.py --flag=name --match_type=match --term_flag=False --get_address=True
#sleep 5


cd ../data_writer/
python queue_to_csv.py --match_type=match --get_address=True



#endtime=$(date +%s)
#cost=$((endtime - starttime))
#echo $cost
#echo '============================='

#cd ../arg_data_pre/
#python execl_data_pre.py

endtime=$(date +%s)
cost=$((endtime - starttime))
echo $cost
echo '============================='

cd ../data_anly/
sh kill_worker.sh



