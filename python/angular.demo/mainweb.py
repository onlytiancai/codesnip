# -*- coding: utf-8 -*-

import web
import os
import json
import shelve
from web.httpserver import StaticMiddleware


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
        return json.dumps(all_houses.values(), indent=4)

    def POST(self):
        house = json.loads(web.data())
        name = house['name'].encode('ascii')
        all_houses[name] = house
        all_houses.sync()
        web.setcookie('name', name, 365 * 24 * 3600)

    def PUT(self):
        house = json.loads(web.data())
        name = house['name'].encode('ascii')
        all_houses[name] = house
        all_houses.sync()


urls = ["/", index,
        "/houses", houses
        ]

app = web.application(urls, globals())
wsgiapp = app.wsgifunc(StaticMiddleware)

if __name__ == '__main__':
    app.run()
