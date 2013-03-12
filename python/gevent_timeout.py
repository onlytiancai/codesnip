# -*- coding: utf-8 -*-

import gevent
from gevent.pool import Pool
from gevent import socket
from collections import defaultdict

pool = Pool(1000)
result_statistics = defaultdict(int)


def connect(ip):
    print 'begin connect %s' % ip
    result_statistics["begin count"] +=  1
    try:
        with gevent.Timeout(1, False):
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.connect((ip, 80))
            client.close()
    except Exception, ex:
        result_statistics["faild count"] +=  1
        print 'end connect %s %s' % (ip, ex)
    else:
        result_statistics["ok count"] +=  1
        print 'end connect %s ok' % ip 

def ip_gen():
    for i in range(1, 256):
        for j in range(1, 256):
            for k in range(1, 256):
                for l in range(1, 256):
                    yield "%s.%s.%s.%s" % (i, j, k, l)

def go():
    for ip in ip_gen():
        pool.spawn(connect, ip)

if __name__ == '__main__':
    gevent.spawn(go)
    gevent.sleep(0)
    pool.join(timeout=10)
    print result_statistics
