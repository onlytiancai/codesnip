# -*- coding: utf-8 -*-

import web
import os
from web.httpserver import StaticMiddleware

curdir = os.path.dirname(__file__)
render = web.template.render(os.path.join(curdir, 'templates/'))


class index(object):
    def GET(self):
        web.header('Content-Type', 'text/html; charset=utf-8', unique=True)
        return render.index()


urls = ["/", index,
        ]

app = web.application(urls, globals())
wsgi_app = app.wsgifunc(StaticMiddleware)

if __name__ == "__main__":
    app.run()
