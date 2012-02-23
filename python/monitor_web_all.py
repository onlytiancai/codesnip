#encoding=utf-8
'''
先用gevent跑，然后把超时的用curl重试
'''
from gevent import spawn,joinall,sleep
from gevent.pool import Pool
from monitor_web_gevent import process_results, curl as gevent_curl
from monitor_web_curl import parse_curl_info
from popentimeout import  runwithtimeout

CONNECT_TIMEOUT,DATA_TIMEOUT = 10, 10
IP_COUNT,POOL_SIZE = 1000, 100

def curl(ip):
    cmd = 'curl -I --connect-timeout %s -m %s %s ' \
            % (CONNECT_TIMEOUT, DATA_TIMEOUT, ip)
    stdout, stderr = runwithtimeout(cmd, CONNECT_TIMEOUT + DATA_TIMEOUT)
    return parse_curl_info(stdout, stderr), ip


def getiplist():
    iplist =(ip.strip() 
        for i, ip 
        in enumerate(open('iplist.txt', 'r'))
        if i < IP_COUNT)
    return list(set(iplist))

def get_results_use_pool():
    iplist = getiplist()
    pool = Pool(POOL_SIZE)
    jobs = [pool.spawn(gevent_curl, ip) for ip in iplist] 
    joinall(jobs)
    return [job.value for job in jobs]

def get_timeout_ips(results):
    timeout_ips = [result[1] for result in results 
        if result[0] == 'gevent timeout']
    return timeout_ips

def retry_timeout_ips(timeout_ips):
    print 'begin retry timeout ips', len(timeout_ips)
    jobs = [spawn(curl, ip) for ip in timeout_ips]
    joinall(jobs)    
    return [job.value for job in jobs]

def merge_results(retry_results, all_results):
    results = retry_results
    notimeout_results = [result for result in all_results 
        if result[0] != 'gevent timeout']
    results.extend(notimeout_results)
    return results 

def init_log(ips):
    for ip in ips:
        log_path = './log/%s' % (ip,)
        with open(log_path, 'a') as f:
            f.write(ip)

def log_results(results):
    for result in results:
        ip = result[1]
        log_path = './log/%s' % (ip,)
        with open(log_path, 'a') as f:
            try:
                int(result[0])
                f.write('*')
            except:
                f.write('#')

def end_log(ips):
 for ip in ips:
    log_path = './log/%s' % (ip,)
    with open(log_path, 'a') as f:
        f.write('\n')

def process_ips(ips):
    all_results = get_results_use_pool()
    timeout_ips = get_timeout_ips(all_results)
    retry_results = retry_timeout_ips(timeout_ips)
    return merge_results(retry_results, all_results)

if __name__ == '__main__':
    iplist = getiplist()
    init_log(iplist)
    for i in range(100):
        results = process_ips(iplist)
        process_results(results)
        log_results(results)
        print 'sleep.....'
        sleep(30)
    end_log(iplist)
