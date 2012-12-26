from gevent import monkey
monkey.patch_all()
import gevent
from gevent.pool import Pool
import urllib3

http = urllib3.PoolManager()
pool = Pool(10)

def test(name):
    r = http.request('GET', 'http://192.168.1.119:8880/'+name)
    print r.status, r.data

jobs = [pool.spawn(test, str(i)) for i in range(10000)]
gevent.joinall(jobs)
