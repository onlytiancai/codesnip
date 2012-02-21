#encoding=utf-8
'''
使用线程和urllib探测一组IP的可访问性
'''
from Queue import Queue
from threading import Thread
import urllib2
import socket

CONNECT_TIMEOUT,DATA_TIMEOUT = 10, 10
IP_COUNT,POOL_SIZE = 1000, 100
global_results = []

class Worker(Thread):
    """Thread executing tasks from a given tasks queue"""
    def __init__(self, tasks):
        Thread.__init__(self)
        self.tasks = tasks
        self.daemon = True
        self.start()
    
    def run(self):
        while True:
            func, args, kargs = self.tasks.get()
            try:func(*args, **kargs)
            except Exception, e: print e
            self.tasks.task_done()

class ThreadPool:
    """Pool of threads consuming tasks from a queue"""
    def __init__(self, num_threads):
        self.tasks = Queue(num_threads)
        for _ in range(num_threads): Worker(self.tasks)

    def add_task(self, func, *args, **kargs):
        """Add a task to the queue"""
        self.tasks.put((func, args, kargs))

    def wait_completion(self):
        """Wait for completion of all the tasks in the queue"""
        self.tasks.join()


def curl(ip):
    '''
    使用urllib2探测IP是否可以访问，并抽取应答码
    和错误原因
    '''
    url = 'http://' + ip
    request = urllib2.Request(url=url)
    reason, other = None, 0

    try:
        rsp = urllib2.urlopen(request, timeout=CONNECT_TIMEOUT + DATA_TIMEOUT)
        reason, other = rsp.getcode(), rsp.msg
    except urllib2.HTTPError, ex:
        reason, other = ex.code, ex.msg 
    except urllib2.URLError, ex:
        reason = ex.reason
        if isinstance(reason, socket.timeout):
            reason = reason.message
        elif isinstance(reason, socket.error):
            reason = reason.strerror 
    finally:
        print reason, ip, other
        result = reason, ip, other 
        global_results.append(result)

def process_results(results):
    '''
    处理扫描结果，对结果进行排序并打印，
    及统计各种结果的数量
    '''
    results = sorted(results)
    stats = {}
    for result in results:
        error = result[0]
        stats.setdefault(error, 0)
        stats[error] = stats[error] + 1
        print result
    
    keys = sorted(stats.keys())
    for key in keys:
        print key, stats[key] 

if __name__ == '__main__':
    iplist  = (ip.strip() 
        for i, ip 
        in enumerate(open('iplist.txt', 'r'))
        if i < IP_COUNT)

    pool = ThreadPool(POOL_SIZE)
    for ip in iplist:
        pool.add_task(curl, ip)
    pool.wait_completion()
    process_results(global_results)
'''
None 2
200 295
301 7
304 6
400 211
401 7
403 78
404 53
500 4
502 8
503 5
504 3
Connection refused 24
Connection reset by peer 49
No route to host 14
Temporary failure in name resolution 1
timed out 233
'''
