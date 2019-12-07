#!/usr/bin/env python
#-*- coding: utf-8 -*-

import time
import socket
from multiping import MultiPing
import rrd
import settings

def ping(addrs, timeout=1):
    with socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_ICMP) as sock:
        mp = MultiPing(addrs, sock=sock, ignore_lookup_errors=True)
        mp.send()
        responses, no_responses = mp.receive(timeout=timeout)

        for addr, rtt in responses.items():
            print("%s responded in %d ms" % (addr, rtt*1000))
            rrd.rrd_init_or_update('%s.rrd' % addr, int(rtt*1000))
        if no_responses:
            for addr in no_responses:
                rtt = timeout*1000
                print("%s responded in %d ms" % (addr, rtt))
                rrd.rrd_init_or_update('%s.rrd' % addr, rtt)

while True:
    try:
        ping(settings.addrs)
        print('*'*10)
    except Exception as ex:
        print(ex)
        raise
    time.sleep(3)
