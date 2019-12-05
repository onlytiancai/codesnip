#!/usr/bin/env python
#-*- coding: utf-8 -*-

import time
import socket
from multiping import MultiPing

def ping(addrs, timeout=1):
    with socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_ICMP) as sock:
        mp = MultiPing(addrs, sock=sock, ignore_lookup_errors=True)
        mp.send()
        responses, no_responses = mp.receive(timeout=timeout)

        for addr, rtt in responses.items():
            print("%s responded in %d ms" % (addr, rtt*1000))
        if no_responses:
            for addr in no_responses:
                print("%s responded in %d ms" % (addr, timeout*1000))

addrs = ['cloud.tencent.com',
           'aliyun.com',
           'ihuhao.com',
           'baidu.com',
           'stackoverflow.com',
           'cloud.ihuhao.com',
          ]

while True:
    try:
        ping(addrs)
        print('*'*10)
    except Exception as ex:
        print(ex)
    time.sleep(1)

