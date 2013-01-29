# -*- coding: utf-8 -*-
from gevent import pywsgi


def hello_world(env, start_response):
    start_response('200 OK', [('Content-Type', 'text/html')])
    return ["<b>hello world</b>"]

server = pywsgi.WSGIServer(('0.0.0.0', 7000), hello_world, log=open('/dev/null', 'w'))
server.serve_forever()
