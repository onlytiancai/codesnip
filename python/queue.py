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

@atexit.register
def goodbye():
    data_list_lock.acquire()
    print('consumer data is:%s' % consumer_data_list)
    print('min consumer data is:%s' % min(consumer_data_list))
    data_list_lock.release()

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

def producer_thread(args):
    logging.info("Thread producer: starting")
    for i in range(100000):
        logging.info("Thread producer: put %d", i)
        q.enqueue(i)

def consumer_thread(args):
    consumer_id = args
    while True:
        data = q.dequeue()
        data_list_lock.acquire()
        consumer_data_list[consumer_id] = data
        data_list_lock.release()
        time.sleep(random.uniform(0.5,2))
        logging.info("Thread consumer[%d]: %d", consumer_id, data)

if __name__ == '__main__':
    producer = threading.Thread(target=producer_thread, args=(None,))
    consumers = [threading.Thread(target=consumer_thread, args=(i,)) for i in range(MAX_CONSUMERS)]
    producer.start()
    for t in consumers:
        t.start()
    producer.join()
    for t in consumers:
        t.join()
