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
    
    parser.add_argument('--file_path', type=str, default='/home/zhijiehuang/github/data/sex_age/',
                        help='gevent.')
    
    parser.add_argument('--tmp_data_path', type=str, default='/data/sex_age/',
                        help='path to QuanSongCi.txt')
    
    parser.add_argument('--host', type=str, default='localhost',
                        help='localhost.')
    
    parser.add_argument('--user', type=str, default='root',
                        help='user.')
    
    parser.add_argument('--passwd', type=str, default='root',
                        help='passwd.')
    
    parser.add_argument('--port', type=int, default=3306,
                        help='port.')
    
    parser.add_argument('--db', type=str, default='sex_age',
                        help='db.')
    
    parser.add_argument('--del_maxmin_mod', type=bool, default=True,
                        help='db.')
    
    parser.add_argument('--del_pca_mod', type=bool, default=True,
                        help='pca.')

    parser.add_argument('--pca_rate', type=float, default=0.7,
                        help='pca.')

    parser.add_argument('--t1_feature', type=str, default='32,33,36,43,42,31,17,19,4',
                        help='d')

    parser.add_argument('--t2_feature', type=str, default='132,11,124,251,223,94,158,83,246,187,218,103,159,37,200,243,102',
                        help='d')
                        
    FLAGS, unparsed = parser.parse_known_args()

    return FLAGS, unparsed


def win_parse_args(check=True):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--cwd', type=str, default='d:/GitHub/sex_age/arg_data_pre',
                        help='path to  tool.')
    
    parser.add_argument('--file_path', type=str, default='F:/GitHub/data/sex_age/',
                        help='gevent.')
    
    parser.add_argument('--host', type=str, default='172.18.18.175',
                        help='localhost.')
    
    parser.add_argument('--user', type=str, default='root',
                        help='user.')
    
    parser.add_argument('--passwd', type=str, default='root',
                        help='passwd.')
    
    parser.add_argument('--port', type=int, default=3306,
                        help='port.')
    
    parser.add_argument('--db', type=str, default='sex_age',
                        help='db.')
    
    parser.add_argument('--t1_feature', type=str, default='32,33,36,43,42,31,17,19,4',
                        help='d')

    parser.add_argument('--t2_feature', type=str, default='132,11,124,251,223,94,158,83,246,187,218,103,159,37,200,243,102',
                        help='d')
    
    FLAGS, unparsed = parser.parse_known_args()

    return FLAGS, unparsed

if pad=='win':
    FLAGS, unparsed = win_parse_args()
elif pad=='linux':
    FLAGS, unparsed = parse_args()

#print(FLAGS.t2_feature.replace('\'','').split(','))


