#encoding=utf-8
'''
演示如何多进程的使用gevent,
1、gevent和multiprocessing组合使用会有很多问题，
  所以多进程直接用subprocess.Popen,进程间不通过fork共享
  任何数据,完全独立运行,并通过socket通信
2、进程间同步不能用multiprocessing.Event,
  因为wait()的时候会阻塞住线程，其它协程的代码无法执行，也
  不能使用gevent.event.Event()，因为它通过multiprocessing.Process
  共享到子进程后，在父进程set()，子进程wait()是不会收到信号的
3、子进程内不能通过signal.signal(signal.SIGINT, signal.SIG_IGN)
  忽略ctrl+c，所以启动主进程时如果没设置后台运行，在ctrl+c时，主进程
  和子进程都会中止而不能优雅退出
4、主进程和子进程的通信和同步使用gevent.socket来实现，子进程收到
  主进程断开连接事件(接受到零字节数据)时,自己优雅退出,相当于主进程
  发消息告诉子进程让子进程退出
5、主进程启动时直接在后台运行，使用"nohup gevent-multil-process.py &"来运行，
  测试时可不用nohup命令，停止主进程时使用kill pid的方式，在主进程里
  会拦截SIGTERM信号，通知并等待子进程退出
'''
import gevent
import gevent.socket as socket
from gevent.event import Event
import os
import sys
import subprocess
import signal

url = ('localhost', 8888)

class Worker(object):
    '''
    子进程运行的代码,通过起一个协程来和主进程通信
    包括接受任务分配请求，退出信号(零字节包)，及反馈任务执行进度
    然后主协程等待停止信号并中止进程(stop_event用于协程间同步)。
    '''
    def __init__(self, url):
        self.url = url
        self.stop_event = Event()
        gevent.spawn(self.communicate)
        self.stop_event.wait()
        print 'worker(%s):will stop' % os.getpid()
    def exec_task(self, task):
        print 'worker(%s):execute task:%s' % (os.getpid(), task.rstrip('\n'))
    def communicate(self):
        print 'worker(%s):started' % os.getpid()
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect(self.url)
        fp = client.makefile()
        while True:
            line = fp.readline()
            if not line:
                self.stop_event.set()
                break
            '单独起一个协程去执行任务，防止通信协程阻塞'
            gevent.spawn(self.exec_task, line)

class Master():
    '''
    主进程运行代码,启动单独协程监听一个端口以供子进程连接和通信用，
    通过subprocess.Popen启动CPU个数个子进程,注册SIGTERM信号以便在
    KILL自己时通知子进程退出，主协程等待停止事件并退出主
    '''
    def __init__(self, url):
        self.url = url
        self.workers = []
        self.stop_event = Event()

        gevent.spawn(self.communicate)
        gevent.sleep(0) #让communicate协程有机会执行，否则子进程会先启动

        self.process = [subprocess.Popen(('python',sys.argv[0],'worker'))
            for i in xrange(3)] #启动multiprocessing.cpucount-1个子进程

        gevent.signal(signal.SIGTERM, self.stop) #拦截kill信号

        gevent.spawn(self.test) #测试分发任务

        self.stop_event.wait() 

    def communicate(self):
        print 'master(%s):started' % os.getpid()
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(url)
        server.listen(1024)
        while True:
            worker, addr = server.accept()
            print 'master(%s):new worker' % os.getpid()
            self.workers.append(worker)

    def stop(self):
        print 'master stop'
        for worker in self.workers:
            worker.close()
        for p in self.process:
            p.wait()
        self.stop_event.set()

    def test(self):
        import random
        while True:
            if not self.workers:
                gevent.sleep(1)
                continue
            task = str(random.randint(100,10000))
            worker = random.choice(self.workers)
            worker.send(task)
            worker.send('\n')
            gevent.sleep(1)

if len(sys.argv) == 1:
    Master(url)
else:
    Worker(url)
