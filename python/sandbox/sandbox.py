# -*- coding: utf-8 -*-

import os
import pwd

# 必须放在chroot前面，否则会报错
import socket
import time
import subprocess

# 必须放在chroot前面，否则会报错
uid = pwd.getpwnam('pythonapp').pw_uid

# 必须放在setuid前面，否则会报错
os.chroot("./")
print "Changed root path successfully!!"

os.setuid(uid)

# 预期只显示当前目录下的文件
ret = os.listdir('/')
print 'os.listdir: %s' % ','.join(ret)

# 预期写入成功
with open('./log/log.log', 'w') as f:
    f.write('test\n')

ret = subprocess.check_output('ls /', shell=True) 
print 'subprocess', ret

# 预期监听成功
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('0.0.0.0', 12345))
s.listen(1)

time.sleep(100000)
