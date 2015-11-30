# -*- coding: utf-8 -*-

import os, sys
import subprocess
os.chroot("./")
print "Changed root path successfully!!"

ret = os.listdir('/')
print ret
