#!/usr/bin/env python

import os
import pwd

open('/etc/shadow').close()
os.setuid(pwd.getpwnam('nobody').pw_uid)
open('/etc/shadow').close()
