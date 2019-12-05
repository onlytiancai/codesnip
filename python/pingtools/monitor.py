#!/usr/bin/env python
#-*- coding: utf-8 -*-

import time
import socket
from multiping import multi_ping 

def ping(addrs):
    with socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_ICMP) as sock:
        responses, no_response = multi_ping(addrs, sock=sock, timeout=0.5, retry=2,
                                                ignore_lookup_errors=True)

        for addr, rtt in responses.items():
            print("%s responded in %f seconds" % (addr, rtt))

addrs = ['cloud.tencent.com',
           'aliyun.com',
           'baidu.com',
           'stackoverflow.com',
          ]

while True:
    ping(addrs)
    time.sleep(1)

