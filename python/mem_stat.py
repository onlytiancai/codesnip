# -*- coding: utf-8 -*-
'''
crontab设置每隔10分钟记录一次内存耗费Top 10

*/10 * * * * cd /root/src/down-monitor/monitorv2/stuff && python mem_stat.py >>/var/log/mem_stat.log
'''

import psutil
from datetime import datetime

print '*' * 20, datetime.now()

processes = psutil.get_process_list()
ps = [(p.get_memory_info().vms, ' '.join(p.cmdline)) 
      for p in processes
      if p.name]

for p in sorted(ps,reverse=True)[:10]:
    print p
