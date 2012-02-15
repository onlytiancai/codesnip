#encoding=utf-8
'''
使用gevent协程池和gevent.monkey patch过的urllib2
探测一组IP的可访问性
'''
from gevent import joinall,Timeout
from gevent import monkey;monkey.patch_all()
from gevent.pool import Pool
import urllib2
import socket

CONNECT_TIMEOUT,DATA_TIMEOUT = 10, 10
IP_COUNT,POOL_SIZE = 1000, 100

def curl(ip):
    '''
    使用urllib2探测IP是否可以访问，并抽取应答码
    和错误原因
    '''
    url = 'http://' + ip
    request = urllib2.Request(url=url)
    reason, other = None, 0

    timeout = Timeout(CONNECT_TIMEOUT + DATA_TIMEOUT)
    timeout.start()
    try:
        rsp = urllib2.urlopen(request)
        reason, other = rsp.getcode(), rsp.msg
    except Timeout, t:
        if t is timeout:
            reason, other = 'gevent timeout', 0
        else:
            reason, other= 'gevent timeout 2', 0
    except urllib2.HTTPError, ex:
        reason, other = ex.code, ex.msg 
    except urllib2.URLError, ex:
       reason = ex.reason
       if isinstance(reason, socket.timeout):
           reason = reason.message
       elif isinstance(reason, socket.error):
           reason = reason.strerror 
    finally:
        timeout.cancel()
        print reason, ip, other
        return reason, ip, other 

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

    pool = Pool(POOL_SIZE)
    jobs = [pool.spawn(curl, ip) for ip in iplist] 
    joinall(jobs)
    results = [job.value for job in jobs]
    process_results(results)
'''
扫描结果：
None 1
200 282
301 7
304 6
400 222
401 7
403 84
404 54
500 5
502 9
503 6
504 2
Connection refused 33
Connection reset by peer 48
No route to host 14
gevent timeout 219
unknown 1
'''
