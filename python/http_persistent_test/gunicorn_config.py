import multiprocessing

bind = "192.168.1.119:8880"
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "gevent"
keepalive = 5
