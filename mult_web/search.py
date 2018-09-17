#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import uuid
from lxml import etree
import requests
import random
import sys
import time
import pandas as pd
import numpy as np
import os
import traceback
from subprocess import *
sys.path.append('..')
from flags import FLAGS, unparsed
no_content = "404_no_content"
no_hit = "405_no_hit"
no_response = "401_no_response"
query_timeout = 30
short_name_len = 4
company_title_len = 30



def parse(brand,type_no, url_content):
    """
    process content
    resunt final result
    """
    price = 1200
    sell_date = 20130101
    st = time.time()
    tag_objs = get_all_nodes(brand,type_no, url_content)
    if not tag_objs:
        print(no_hit)
        return False, price,sell_date

    price, sell_date = tag_objs['price'], tag_objs['sell_date']
    time_cost = time.time() - st
    print("parse**** time cost ", time_cost)
    return True, price, sell_date


def get_all_nodes(brand,type_no, data):
    tag_objs = {}
    ed = etree.HTML(data)
#    all = ed.xpath("//*[@class='result c-container ']")
    if ed is  None:
        return tag_objs
    try:
        ae = ed.xpath('//div[@class="c-abstract"]')[0]
        text = ae.xpath('./text()')[0]
        text_list=text.replace('[',' ').replace(']',' ').replace('￥',' ').split(' ')
        price=text_list[2]
        sell_date=text_list[5]
        print(price,sell_date)
        tag_objs['price']=price
        tag_objs['sell_date']=sell_date
    except:
        message = traceback.format_exc()
        print( message)
    return tag_objs




def get_content(brand,type_no):
    st = time.time()
    url = f'http://www.baidu.com/s?wd='+brand+' '+type_no+' 上市'
    agent_list = [
        "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; AcooBrowser; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
        "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.0; Acoo Browser; SLCC1; .NET CLR 2.0.50727; Media Center PC 5.0; .NET CLR 3.0.04506)",
        "Mozilla/4.0 (compatible; MSIE 7.0; AOL 9.5; AOLBuild 4337.35; Windows NT 5.1; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
        "Mozilla/5.0 (Windows; U; MSIE 9.0; Windows NT 9.0; en-US)",
        "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Win64; x64; Trident/5.0; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 2.0.50727; Media Center PC 6.0)",
        "Mozilla/5.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0; WOW64; Trident/4.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 1.0.3705; .NET CLR 1.1.4322)",
        "Mozilla/4.0 (compatible; MSIE 7.0b; Windows NT 5.2; .NET CLR 1.1.4322; .NET CLR 2.0.50727; InfoPath.2; .NET CLR 3.0.04506.30)",
        "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN) AppleWebKit/523.15 (KHTML, like Gecko, Safari/419.3) Arora/0.3 (Change: 287 c9dfb30)",
        "Mozilla/5.0 (X11; U; Linux; en-US) AppleWebKit/527+ (KHTML, like Gecko, Safari/419.3) Arora/0.6",
        "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.8.1.2pre) Gecko/20070215 K-Ninja/2.1.1",
        "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN; rv:1.9) Gecko/20080705 Firefox/3.0 Kapiko/3.0",
        "Mozilla/5.0 (X11; Linux i686; U;) Gecko/20070322 Kazehakase/0.4.5",
        "Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.0.8) Gecko Fedora/1.9.0.8-1.fc10 Kazehakase/0.5.6",
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_3) AppleWebKit/535.20 (KHTML, like Gecko) Chrome/19.0.1036.7 Safari/535.20",
        "Opera/9.80 (Macintosh; Intel Mac OS X 10.6.8; U; fr) Presto/2.9.168 Version/11.52",
    ]
    retry_times = 2
    while retry_times:
        try:
            query_header = {
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
                "Accept-Encoding": "gzip, deflate, br",
                "Accept-Language": "zh-CN,zh;q=0.9",
                "Cache-Control": "max-age=0",
                "User-Agent": random.choice(agent_list),
            }
            query_cookie = {
                "BAIDUID": "%s:FG=1" % str(uuid.uuid4()).replace('-', '').upper(),
                "BD_CK_SAM": "1",
                "PSTM": "%s" % (int(time.time())),
                "PSINO": "1",
            }
            resp = requests.get(url,
                                headers=query_header,
                                timeout=10,
                                cookies=query_cookie,
                                )
            if resp.status_code == 200:
                print("----------get_content cost: ", time.time()-st, retry_times)
                return parse(brand,type_no, resp.text)
        except Exception as e:
            print(f"faild to get content: {url}, exceptions: {e}, ", '')
        retry_times -= 1
    print("=========get_content cost: ", time.time() - st, retry_times)
    return False,'', url


if __name__ == "__main__":
    deviceid_brand=pd.read_csv(FLAGS.file_path+'new_deviceid_brand.csv')
    ret=get_content('samsung','gt-i9507v')
    print(ret)