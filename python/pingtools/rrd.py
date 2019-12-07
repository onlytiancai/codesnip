# coding:utf-8
import os
import time

import rrdtool
from settings import BASE_DIR


def rrd_init_or_update(rrdname, rtt):
    base_dir = os.path.join(BASE_DIR, "rrddatas")
    rrdpath = os.path.join(base_dir, rrdname)
    if not os.path.isdir(base_dir):
        os.makedirs(base_dir)
    if os.path.isfile(rrdpath):
        rrd_update(rrdpath, rtt)
    else:
        rrd_init(rrdpath)
        rrd_update(rrdpath, rtt)


def rrd_init(rrdname):
    """
    聚合时间根据自己需要
    """
    rrdtool.create(rrdname,
                   "--start", str(int(time.time()) - 3600*24),
                   "--step", '10',
                   "DS:rtt:GAUGE:20:0:5000",
                   "RRA:AVERAGE:0.5:1:8640", # 10 秒存 1 天
                   "RRA:AVERAGE:0.5:6:4320", # 1 分钟存 3 天 
                   "RRA:AVERAGE:0.5:360:168", # 1 小时存 7 天
                   "RRA:AVERAGE:0.5:8640:797", # 1 天 存 2 年 
                   "RRA:MAX:0.5:1:8640",
                   "RRA:MAX:0.5:6:4320",
                   "RRA:MAX:0.5:360:168",
                   "RRA:MAX:0.5:8640:797",
                   "RRA:MIN:0.5:1:8640",
                   "RRA:MIN:0.5:6:4320",
                   "RRA:MIN:0.5:360:168",
                   "RRA:MIN:0.5:8640:797")


def rrd_update(rrdname, rtt):
    rrdtool.update(rrdname, "N:%s" % (rtt))


if __name__ == '__main__':
    rrd_init_or_update('test.rrd', 123.4)
