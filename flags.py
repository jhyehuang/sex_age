#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import datetime

import pytz

FLAGS, unparsed='',''

#pad='win'
#pad='tiny'
pad='linux'

tz = pytz.timezone('Asia/Shanghai')
current_time = datetime.datetime.now(tz)

def parse_args(check=True):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--cwd', type=str, default='/home/zhijiehuang/github/sex_age/arg_data_pre',
                        help='path to  tool.')
    
    parser.add_argument('--host', type=str, default='localhost',
                        help='localhost.')
    
    parser.add_argument('--user', type=str, default='root',
                        help='user.')
    
    parser.add_argument('--passwd', type=str, default='passwd',
                        help='passwd.')
    
    parser.add_argument('--port', type=int, default=3306,
                        help='port.')
    
    parser.add_argument('--db', type=str, default='sex_age',
                        help='db.')
    
    FLAGS, unparsed = parser.parse_known_args()

    return FLAGS, unparsed


def win_parse_args(check=True):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--cwd', type=str, default='d:/GitHub/sex_age/arg_data_pre',
                        help='path to  tool.')
    
    parser.add_argument('--file_path', type=str, default='d:/GitHub/data/sex_age/',
                        help='gevent.')
    
    parser.add_argument('--host', type=str, default='172.18.18.175',
                        help='localhost.')
    
    parser.add_argument('--user', type=str, default='root',
                        help='user.')
    
    parser.add_argument('--passwd', type=str, default='passwd',
                        help='passwd.')
    
    parser.add_argument('--port', type=int, default=3306,
                        help='port.')
    
    parser.add_argument('--db', type=str, default='sex_age',
                        help='db.')
    
    FLAGS, unparsed = parser.parse_known_args()

    return FLAGS, unparsed

if pad=='win':
    FLAGS, unparsed = win_parse_args()
elif pad=='linux':
    FLAGS, unparsed = parse_args()


