# -*- coding:utf-8 -*-
'''
在Python应用里发送业务指标到自定义监控

- 对每分钟请求数之类的指标用`c.increment()`
- 对平均响应时间之类的指标用`c.avg_increment()`
- 对当前在线人数之类的指标用`c.setraw(1024)`

使用方法如下

    # 定义计数器，启动发送线程
    cm = CounterManager(api_key='test_key', host='test_host')
    c = Counter('requests_count', cm)
    cm.start()

    # 在收到请求时递增计数器
    c.increment()
'''
try:
    from gevent import monkey
    monkey.patch_time()
    monkey.patch_thread()
    monkey.patch_socket()
except ImportError:
    pass

import time
import socket
import threading
import logging


class CounterManager(threading.Thread):
    def __init__(self, api_key, host=socket.gethostname(), ip='127.0.0.1',
                 interval=60, collecter_url=('collector.monitor.dnspod.cn', 2003)):

        threading.Thread.__init__(self)
        self.setName('CounterManager:%s' % api_key)
        self.setDaemon(True)

        self.api_key = api_key
        self.host = self._replace_valid_char(host)
        self.ip = self._replace_valid_char(ip)
        self.interval = interval
        self.collecter_url = collecter_url
        self.counters = []
        self.client = socket.socket()
        self.client.connect(collecter_url)

    def _replace_valid_char(self, input):
        input = input.replace('/', '_')
        input = input.replace(' ', '_')
        return input

    def add_counter(self, counter):
        logging.info('add counter:%s', counter)
        self.counters.append(counter)
        counter.api_key = self.api_key
        counter.host = self.host
        counter.ip = self.ip

    def send_data_to_center(self, counter):
        to_send_data = counter._get_data()
        if to_send_data:
            logging.info('send_data_to_center:%s', to_send_data)
            self.client.send(to_send_data)

    def run(self):
        logging.info('counter_manager start:%s', self.api_key)
        while True:
            try:
                for counter in self.counters:
                    self.send_data_to_center(counter)
            except:
                logging.exception('send_data_to_center error')
                try:
                    self.client.connect(self.collecter_url)
                except:
                    logging.exception('auto reconnect error')
            finally:
                time.sleep(self.interval)

    def __del__(self):
        try:
            self.client.close()
        except:
            pass


class Counter(object):
    '计数器'
    def __init__(self, metric_name, cm, auto_clear=1):
        '''创建一个计数器
        metric_name是计数器名称
        auto_clear表示,是否发送给collecter后清零计数器,默认是清零
        '''
        self.metric_name = metric_name
        self.value = 0
        self.has_data = False
        self.auto_clear = auto_clear
        self.count = 0
        cm.add_counter(self)

    def _pre_set_value(self):
        self.count += 1
        self.has_data = True

    def increment(self, value=1):
        self._pre_set_value()
        self.value += value

    def decrement(self, value=1):
        self._pre_set_value()
        self.value -= value

    def avg_increment(self, value=1):
        self._pre_set_value()
        self.value = (self.value + value) / self.count

    def setraw(self, value=1):
        self._pre_set_value()
        self.value = value

    def reset(self):
        self.value = 0
        self.count = 0
        self.has_data = False

    def _get_data(self):
        if not self.has_data:
            return None

        data = '%(api_key)s/%(host)s/%(ip)s/%(metric_name)s %(value)s %(timestamp)s\n'
        self.timestamp = int(time.time())
        data = data % self.__dict__

        if self.auto_clear:
            self.reset()

        return data

    def __str__(self):
        return self.metric_name


def test_default():
    '''
    pip install nosetests
    pip install mock
    nosetests counter.py --nologcapture -s
    '''
    from mock import Mock
    globals()['socket'] = Mock()
    time.time = Mock(return_value=123456)

    logging.basicConfig(level=logging.NOTSET)
    cm = CounterManager(api_key='test_key', host='test_host')
    c = Counter('test', cm)

    c.increment()
    assert c.value == 1

    c.decrement()
    assert c.value == 0

    c.reset()
    c.avg_increment(2)
    c.avg_increment(4)
    assert c.value == 3

    c.setraw(5)
    assert c.value == 5
    
    cm.send_data_to_center(c)
    expect = 'test_key/test_host/127.0.0.1/test 5 123456'
    cm.client.send.assert_called_with(expect)
