# -*- coding: utf-8 -*-

import web
import os
import json
from web.httpserver import StaticMiddleware
from datetime import datetime
import uuid
import logging
import db
import urllib

logging.basicConfig(filename=os.path.join(os.getcwd(), 'log.log'), level=logging.DEBUG)


curdir = os.path.dirname(__file__)
render = web.template.render(os.path.join(curdir, 'templates/'))


class index(object):
    def GET(self):
        web.header('Content-Type', 'text/html; charset=utf-8', unique=True)
        return render.index()


class houses(object):
    def GET(self):
        web.header('Content-Type', 'application/json; charset=utf-8', unique=True)
        all_houses = db.get_all_houses()
        return json.dumps(all_houses, indent=4)

    def POST(self):
        house = json.loads(web.data())
        logging.debug('add houses:%s %s', web.ctx.ip, house)
        token = str(uuid.uuid1())
        db.add_house(web.ctx.ip,
                     house['name'].strip(),
                     house['text'],
                     datetime.now().strftime('%Y-%m-%d %H:%M'),
                     token)
        web.setcookie('name', urllib.quote(house['name'].encode('utf-8')), 10 * 365 * 24 * 3600)
        web.setcookie('token', token, 10 * 365 * 24 * 3600)

    def PUT(self):
        house = json.loads(web.data())
        logging.debug('edit houses:%s %s', web.ctx.ip, house)
        token = web.cookies().get('token')
        db.modify_house(web.ctx.ip,
                     house['name'].strip(),
                     house['text'],
                     datetime.now().strftime('%Y-%m-%d %H:%M'),
                     token)

class history(object):
    def GET(self, name):
        web.header('Content-Type', 'application/json; charset=utf-8', unique=True)
        all_houses = db.get_history(name)
        return json.dumps(all_houses, indent=4)


        
urls = ["/", index,
        "/houses", houses,
        "/history/([^/]+)", history 
        ]

app = web.application(urls, globals())
wsgiapp = app.wsgifunc(StaticMiddleware)

if __name__ == '__main__':
    app.run()
