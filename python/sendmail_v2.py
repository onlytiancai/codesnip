import smtplib
from email.mime.text import MIMEText
from configparser import ConfigParser
conf = ConfigParser()
conf.read('config.ini')

msg = MIMEText('Have a nice day guy!')
msg['Subject'] = 'Hi, Nice to meet you!'
msg['From'] = 'wawasoft@qq.com'
msg['To'] = 'hao@ihuhao.com'

with smtplib.SMTP('smtp.qq.com', 587) as server:
    server.starttls()
    server.login(conf['smtp']['user'], conf['smtp']['pass'])
    server.send_message(msg)

