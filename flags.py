#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import datetime

import pytz

FLAGS, unparsed='',''

tz = pytz.timezone('Asia/Shanghai')
current_time = datetime.datetime.now(tz)

def parse_args(check=True):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--process', type=int, default=8,
                        help='path to my tool.')
    
    parser.add_argument('--task_size', type=int, default=20,
                        help='gevent.')
    
    parser.add_argument('--SEARCH_URL', type=str, default="http://172.18.52.171:9200",
                        help='path to save log and checkpoint.')
    
    parser.add_argument('--match_type', type=str, default='term',
                        help='path to save train test and .')

    parser.add_argument('--flag', type=str, default='tel',
                        help='flag')
    
    parser.add_argument('--top_n', type=int, default=1,
                        help='top_n')

    parser.add_argument('--term_flag', type=bool, default=False,
                        help='控制精准查到以后是否继续模糊')

    parser.add_argument('--get_address', type=bool, default=False,
                        help='是否需要地址')
    
    FLAGS, unparsed = parser.parse_known_args()

    return FLAGS, unparsed


FLAGS, unparsed = parse_args()

