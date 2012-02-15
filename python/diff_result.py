#encoding=utf-8
'''
处理 monitor_web_curl.py, monitor_web_gevent.py的结果
找出gevent报错，但curl没报错的ip
'''
def get_result_error(f):
    for line in open(f):
        if not line.startswith('('):continue 
        arr = line.split(',')
        if len(arr) < 2:continue
        info, ip = arr[0].strip("()'\"\n? "), arr[1].strip("()'\"\n? ")
        if info.isdigit():continue
        yield info, ip 

gevent_error = get_result_error('result_gevent.txt')
curl_error = get_result_error('result_curl.txt')
set1 = set(ip for info, ip in gevent_error)
set2 = set(ip for info, ip in curl_error)

diff = set1.difference(set2)
gevent_error = get_result_error('result_gevent.txt')
d = [(ip,info) for info, ip in gevent_error if ip in diff]
print len(set1), len(set2), len(diff), len(d)

for ip, info in d:
    print ip, info
 

