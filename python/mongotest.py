# -*- coding:utf-8 -*-
'''
参考链接：
http://stackoverflow.com/questions/11498734/best-practice-of-django-pymongo-pooling
http://stackoverflow.com/questions/7166998/pymongo-gevent-throw-me-a-banana-and-just-monkey-patch
http://api.mongodb.org/python/current/api/pymongo/connection.html#pymongo.connection.Connection.start_request
'''
from gevent import monkey;monkey.patch_all()
import pymongo
from datetime import datetime
import random
import gevent

conn = pymongo.Connection(auto_start_request=False)
db = conn.test

def gen_test_data():
    db.temp.drop()
    t1 = datetime.now()
    for i in range(10*10000):
        print i
        db.temp.insert({'id': random.randint(1,10)})
    print 'gen test data take time:%s' % (datetime.now() - t1)

def test_find_data():
    with conn.start_request():
        t1 = datetime.now()
        results = list(db.temp.find({'id':random.randint(1,10)}))
        print 'test find data take time:%s %s' % (len(results), datetime.now() - t1)
if __name__ == '__main__':
    gen_test_data()
    while True:
        [gevent.spawn(test_find_data) for i in range(10)]
        gevent.sleep(1)
