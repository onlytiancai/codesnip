#encoding=utf-8
'''
一个运行的socket server一般情况下
都有个协程阻塞在socket.accept上
这时候想想停止server,不能用标志位的方式
让它停止，可以用kill给协程发个信号让它退出
'''
import gevent
from gevent import socket

def foo():
    try:
        server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        server.bind(('0.0.0.0',12345))
        server.listen(1024)
        print 'listen:12345'
        while True:
            client,addr = server.accept()
            print 'new conn:%s %s' % addr
            client.send(str(addr))
            client.close()
    except gevent.GreenletExit:
        print 'exit'

task =  gevent.spawn(foo)
gevent.sleep(10)
task.kill()
print 'killed'
