# -*- coding: utf-8 -*-
'''
实验目的：测试单核CPU需不需要加锁

count = count + 1 这个语句不是一个原子的操作，多个线程/协程在不加锁的情况会造成竞争条件。

但实际结果是：
1. 用python自带的线程测试，出现竞争条件问题，多次测试每次结果都不是5000000
2. 用gevent的协程测试，也没有问题，最终用结果都是5000000

结论：

1. 尽管python有GIL，同一时间只有一个线程执行，但不用锁是不安全的。
2. gevent的协程可能实现的有问题，不会再一条语句执行半拉时做上下文切换，所以不加锁也不会有问题。
3. 可能好些语言的协程也做不到像真正的操作系统线程那样抢占式，而是一个函数执行完才会做上下文切换。

'''
import threading

use_gevent = False       # 是否使用gevent
use_debug = False        # 是否打印测试输出
cycles_count = 100*10000 # 每个计数器线程/协程循环增加计数器的次数


if use_gevent:
    from gevent import monkey
    monkey.patch_thread()

count = 0


class Counter(threading.Thread):
    def __init__(self, name):
        self.thread_name = name
        super(Counter, self).__init__(name=name)

    def run(self):
        global count
        for i in xrange(cycles_count):
            if use_debug:
                print '%s:%s' % (self.thread_name, count)
            count = count + 1

counters = [Counter('thread:%s' % i) for i in range(5)]
for counter in counters:
    counter.start()
for counter in counters:
    counter.join()

print 'count=%s' % count
