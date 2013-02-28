# -*- coding: utf-8 -*-

import web

app_jslink = '<script src="/static/sea-modules/seajs/1.3.0/sea-debug.js" data-main="/dnspod/static/main"></script>'
app_desc = 'DNSPod'


class Api(object):
    def POST(self):
        pass


urls = ["/Api", Api,
        ]

app = web.application(urls, globals())

if __name__ == "__main__":
    app.run()
