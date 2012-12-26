from gevent import monkey
monkey.patch_all()
import urllib3

http = urllib3.PoolManager()

for i in range(100):
    r = http.request('GET', 'http://192.168.1.119:8880/hello/mm')
    print r.status, r.data
