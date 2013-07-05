# -*- coding: utf-8 -*-

import web
import os
import json
import shelve
from web.httpserver import StaticMiddleware
import logging


curdir = os.path.dirname(__file__)
render = web.template.render(os.path.join(curdir, 'templates/'))

all_houses = shelve.open('houses.db')


class index(object):
    def GET(self):
        web.header('Content-Type', 'text/html; charset=utf-8', unique=True)
        return render.index()


class houses(object):
    def GET(self):
        web.header('Content-Type', 'application/json; charset=utf-8', unique=True)
        all_houses.sync()
        return json.dumps(all_houses.values(), indent=4)

    def POST(self):
        house = json.loads(web.data())
        del house['ip']
        name = house['name'].strip().encode('ascii')
        house['ip'] = web.ctx.ip
        all_houses[name] = house
        all_houses.sync()
        web.setcookie('name', name, 365 * 24 * 3600)

    def PUT(self):
        house = json.loads(web.data())
        del house['ip']
        name = house['name'].strip().encode('ascii')
        dbhouse = all_houses.get(name)
        if dbhouse is None: return 
        
        # 只有添加时的IP可以修改
        if dbhouse.get('ip', '') in ('', web.ctx.ip):
            house['ip'] = web.ctx.ip
            all_houses[name] = house
            all_houses.sync()
        else:
            print '非法修改', web.ctx.ip, name


urls = ["/", index,
        "/houses", houses
        ]

app = web.application(urls, globals())
wsgiapp = app.wsgifunc(StaticMiddleware)

if __name__ == '__main__':
    app.run()
