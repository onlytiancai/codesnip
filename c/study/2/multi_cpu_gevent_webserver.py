# -*- coding: utf-8 -*-
from gevent.server import _tcp_listener
from gevent import pywsgi
from gevent.monkey import patch_all; patch_all()
from multiprocessing import Process, cpu_count


def hello_world(env, start_response):
    start_response('200 OK', [('Content-Type', 'text/html')])
    return ["<b>hello world</b>"]

listener = _tcp_listener(('0.0.0.0', 7000))


def serve_forever(listener):
    pywsgi.WSGIServer(listener, hello_world, log=open('/dev/null', 'w')).serve_forever()

for i in range(cpu_count() - 1):
    Process(target=serve_forever, args=(listener,)).start()

serve_forever(listener)
