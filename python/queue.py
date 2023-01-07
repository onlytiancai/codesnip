import threading
import collections
import logging
import threading
import time
import random
import atexit

format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO,
                    datefmt="%H:%M:%S")

MAX_CONSUMERS = 10
data_list_lock = threading.RLock()
consumer_data_list = [0 for i in range(MAX_CONSUMERS)]

class BoundedBlockingQueue(object):
    def __init__(self, capacity: int):
        self.pushing = threading.Semaphore(capacity)
        self.pulling = threading.Semaphore(0)
        self.data = collections.deque()
 
    def enqueue(self, element: int) -> None:
        self.pushing.acquire()
        self.data.append(element)
        self.pulling.release()
 
    def dequeue(self) -> int:
        self.pulling.acquire()
        self.pushing.release()
        return self.data.popleft()
        
    def size(self) -> int:
        return len(self.data)

q = BoundedBlockingQueue(100)

def info_thread(args):
    while True:
        data_list_lock.acquire()
        temp = consumer_data_list[:]
        data_list_lock.release()

        logging.info("queue size: %d", q.size())
        logging.info('consumer data is:%s', consumer_data_list)
        logging.info('min consumer data is:%s', min(consumer_data_list))
        time.sleep(1)


def producer_thread(args):
    logging.info("Thread producer: starting")
    for i in range(100000):
        q.enqueue(i)

def consumer_thread(args):
    consumer_id = args
    while True:
        data = q.dequeue()
        data_list_lock.acquire()
        consumer_data_list[consumer_id] = data
        data_list_lock.release()
        time.sleep(random.uniform(0.5,2))

if __name__ == '__main__':
    producer = threading.Thread(target=producer_thread, args=(None,))
    consumers = [threading.Thread(target=consumer_thread, args=(i,)) for i in range(MAX_CONSUMERS)]
    info_t = threading.Thread(target=info_thread, args=(None,))

    producer.setDaemon(True)
    producer.start()
    for t in consumers:
        t.setDaemon(True)
        t.start()
    info_t.start()

    producer.join()
    for t in consumers:
        t.join()
    info_t.start()
