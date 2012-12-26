from gevent import monkey
monkey.patch_all()
import gevent
from gevent.pool import Pool
import requests
session = requests.session()

pool = Pool(10)

def test(name):
    r = session.post('http://172.4.2.20:8880/'+name, data={'name':'girl'})
    print r.status_code, r.text

jobs = [pool.spawn(test, str(i)) for i in range(10000)]
gevent.joinall(jobs)
