#encoding=utf-8
'''
通过进程池和curl探测一组IP的HTTP可访问性
'''
from multiprocessing import Pool
from subprocess import Popen,PIPE
import re
regex_error = re.compile(r'curl: \(\d+\) (.*)\n')
regex_code = re.compile(r'\b(\d\d\d)\b')

CONNECT_TIMEOUT,DATA_TIMEOUT = 10, 10
IP_COUNT, POOL_SIZE = 1000, 100

def curl(ip):
    '''
    调用curl并获取结果，从结果里提取http应答码
    及出错的信息,curl使用了连接超时和获取数据超时两个参数
    '''
    cmd = 'curl -I --connect-timeout %s -m %s %s ' \
            % (CONNECT_TIMEOUT, DATA_TIMEOUT, ip)
    process = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    info = process.stdout.readline().rstrip('\r\n')
    info = regex_code.search(info)
    if info:
        info = info.group(1)

    if not info:
        info = process.stderr.read()
        info = regex_error.search(info)
        if info:
            info= info.group(1)

    print  info, ip
    return info, ip

def process_results(results):
    '''
    处理扫描结果，对结果进行排序并打印，
    及统计各种结果的数量
    '''
    results = sorted(results)
    stats = {}
    for result in results:
        info = result[0]
        stats.setdefault(info , 0)
        stats[info] = stats[info] + 1
        print result

    keys = sorted(stats.keys())
    for key in keys:
        print key, stats[key] 

if __name__ == '__main__':
    iplist  = (ip.strip() 
        for i, ip 
        in enumerate(open('iplist.txt', 'r'))
        if i < IP_COUNT)

    pool = Pool(processes=POOL_SIZE)
    results = pool.map(curl, iplist)
    process_results(results)
'''
扫描结果：
200 227
301 33
302 44
304 6
400 240
401 7
403 88
404 44
406 1
500 4
502 2
503 6
504 2
Empty reply from server 26
Operation timed out after 10 seconds with 0 bytes received 74
connect() timed out! 45
couldn't connect to host 151
'''
