# -*- coding:utf-8 -*-
'《Python Cookbook》 p317'

import os

_proc_status = '/proc/%d/status' % os.getpid()
_scale = {'KB': 1024.0, 'MB': 1024.0 * 1024.0,
          'kB': 1024.0, 'mB': 1024.0 * 1024.0}


def _VmB(VmKey):
    '给定vmKey字符串，返回字节数'
    #获得伪文件/proc/<pid>/status
    try:
        t = open(_proc_status)
        v = t.read()
        t.close()
    except IOError:
        return 0.0  # no-Linux
    
    # 获得VmKey行，如'VmRSS:    9999 kb\n'
    i = v.index(VmKey)
    v = v[i:].split(None, 3)
    if len(v) < 3:
        return 0.0

    return float(v[1]) * _scale[v[2]]


def memory(since=0.0):
    '返回虚拟内存使用的字节数'
    return _VmB('VmSize:') - since


def resident(since=0.0):
    '返回常驻内存使用的字节数'
    return _VmB('VmRSS:') - since


def stacksize(since=0.0):
    '返回栈使用的字节数'
    return _VmB('VmStk:') - since

if __name__ == '__main__':
    def print_momory_statistics():
        print 'VmSize', memory()
        print 'VmRSS', resident()
        print 'VmStk', stacksize()

    def foo():
        return 'hello world' * 10000000

    print_momory_statistics()
    s = foo()
    print_momory_statistics()
