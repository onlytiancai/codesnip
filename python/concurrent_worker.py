'''
多消费者队列
1. 可设置 worker 数限制并发，防止无限并发打垮数据库或下游接口。
2. 可用 CTRL+C 或发送 SIGTERM 优雅停止，正在运行的任务会完成。 
3. 所有任务处理完毕后会自动退出，并打印最后 task_id 以查看进度。

技术细节：
1. 不能强杀 thread，也不能用 daemon 模式，要 join 所有线程，否则可能丢失数据。
2. queue 要用 join 等待队列为空，否则强制退出进程会让内存中的队列数据丢失。
3. 消费者不能用无超时的阻塞 get 任务，否则无法收到主线程的 stop event。
4. 消费者收到 stop event，要再读一次队列，防止队列数据丢失，正常退出不需要。
5. 生产者不能无超时的阻塞 put 任务，否则无法收到 ctrl+c 等信号。
6. 生产者发送完所有任务后，要发送一个 None 任务告诉消费者退出。
'''
import sys
import queue
import signal
import logging
import threading

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("concurrent_worker")
logger.setLevel(logging.DEBUG)

def run_workers(producer, task_func, num_workers):
    stop_event = threading.Event()

    def worker(worker_id, q):
        receive_none = False
        last_task_id = None
        while not stop_event.is_set():
            try:
                args = q.get(timeout=0.5)
                if args is None:
                    receive_none = True
                    q.task_done()
                    logger.info(f'{worker_id} receive None task.')
                    break

                i, data = args
                task_func(worker_id, i, data)
                last_task_id = i
                q.task_done()
            except queue.Empty:
                continue

        logger.info(f'{worker_id} is exiting. receive_none:{receive_none}, qsize:{q.qsize()}')
        if not receive_none:
            try:
                args = q.get(timeout=0.5)
                if args:
                    i, data = args
                    logger.info(f'{worker_id} process last task:{i}')
                    task_func(worker_id, i, data)
                    last_task_id = i
                    q.task_done()
            except queue.Empty:
                pass 

        logger.info(f'{worker_id} exited, last_task_id:{last_task_id}')

    q = queue.Queue(maxsize=num_workers)
    threads = []
    for worker_id in range(num_workers):
        thread = threading.Thread(target=worker, args=(worker_id, q))
        thread.start()
        threads.append(thread)

    def handle_sigterm(signum, frame):
        logger.info("recived SIGTERM, will exit, qsize:%s", q.qsize())
        stop_event.set()
        q.join()
        for thread in threads:
            thread.join()
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_sigterm)

    try:
        for i, data in enumerate(producer):
            if i > 0 and i % 1000 == 0:
                logger.info('progress:%s', i)

            while not stop_event.is_set():
                try:
                    q.put([i, data], timeout=0.5)
                    break
                except queue.Full:
                    continue

        for _ in range(num_workers):
            q.put(None)
        q.join()
    except KeyboardInterrupt:
        logger.info("recived ctrl+c, will exit, qsize:%s", q.qsize())
        stop_event.set()
        q.join()
    finally:
        for thread in threads:
            thread.join()

if __name__ == '__main__':
    import time
    producer = range(100)
    def do_task(worker_id, task_id, data):
        print(worker_id, task_id, data)
        time.sleep(1)

    run_workers(producer, do_task, 10)
