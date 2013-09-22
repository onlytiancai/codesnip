# -*- coding: utf-8 -*-

import smtplib
from email.mime.text import MIMEText

username = 'onlytiancai@163.com'  
password = '111'
me = 'baidu@163.com'
smtphost = 'smtp.163.com'

def send_mail(to, subject, content, ssl=False):
    msg = MIMEText(content)
    msg['Subject'] = subject 
    msg['From'] = me 
    msg['To'] = to 
    print msg.as_string()
      
    server = smtplib.SMTP(smtphost)  
    if ssl:
        server.starttls()  
    server.login(username,password)  
    server.sendmail(me, to, msg.as_string())
    server.quit()

send_mail('41354@qq.com', '这是一封测试邮件 from python', '这是一封测试邮件')
