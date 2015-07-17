# -*- coding: utf-8 -*-

import os
import pwd

# 必须放在chroot前面，否则会报错
import socket
import time

# 必须放在chroot前面，否则会报错
uid = pwd.getpwnam('nobody').pw_uid

# 必须放在setuid前面，否则会报错
os.chroot("./")
print "Changed root path successfully!!"

os.setuid(uid)

# 预期只显示当前目录下的文件
ret = os.listdir('/')
print ret

# 预期写入成功
with open('./log/log.log', 'w') as f:
    f.write('test\n')

# 预期监听成功
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('0.0.0.0', 12345))
s.listen(1)

time.sleep(100000)
