#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
定时检测你的域名的NS是否被篡改，
配置好域名，NS关键字，报警邮件等配置后，在crontab里建立一个规则就可以了。

    * * * * * python ns_checker.py 

'''

import commands
import smtplib
from email.mime.text import MIMEText

# 你的域名
domain = 'baidu.com'
# 你的域名的NS关键字
ns_key = 'baidu'

# 报警邮件配置
mail_username = ''
mail_password = ''
mail_from = ''
mail_to = ''
mail_smtphost = ''
mail_isssl = True


def send_mail(nameservers):
    nameservers = ' '.join(nameservers)
    print '域名NS被篡改', nameservers
    msg = MIMEText('域名NS被篡改:%s' % nameservers)
    msg['Subject'] = '域名NS被篡改:%s' % nameservers
    msg['From'] = mail_from
    msg['To'] = mail_to
      
    server = smtplib.SMTP(mail_smtphost)
    if mail_isssl:
        server.starttls()
    server.login(mail_username, mail_password)
    server.sendmail(mail_from, mail_to, msg.as_string())
    server.quit()


nameservers = commands.getoutput('dig +short ns %s' % domain).splitlines()
if not all(nameserver.find(ns_key) != -1 for nameserver in nameservers):
    send_mail(nameservers)
