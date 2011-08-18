#encoding=utf-8
from gevent import spawn,sleep
from gevent.queue import Queue
from gevent.coros import BoundedSemaphore
import random

_q = Queue()
_bs = BoundedSemaphore(10) #数据库并发最大10
_count = 0
def write_db(req):
    '模拟写DB，随机休眠1-3秒，执行完毕释放信号'
    global _count
    _count += 1
    print 'write %s to db' % req
    sleep(random.randint(1,3)) 
    _bs.release()
    _count -= 1

def read_queue():
    '读队列协程，信号量不为空切队列不为空时执行'
    while True:
        _bs.acquire()
        req = _q.get()
        spawn(write_db, req)
def info():
    while True:
        print 'qsize:%s,concurrent:%s' % (_q.qsize(),_count)
        sleep(1)
if __name__ == '__main__':
    spawn(read_queue)
    spawn(info)
    #模拟每秒接受5-30个请求
    while True:
        for i in xrange(random.randint(5,30)):
            _q.put_nowait(random.randint(0,1000))
        sleep(1)
