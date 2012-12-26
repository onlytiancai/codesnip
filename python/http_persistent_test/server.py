from gevent import monkey
monkey.patch_all()
import gevent
from gevent.pywsgi import WSGIServer
import web
        
urls = (
    '/(.*)', 'hello'
)

app = web.application(urls, globals())
app = app.wsgifunc()

class hello:        
    def GET(self, name):
        gevent.sleep(1)
        if not name: 
            name = 'World'
        return 'Hello, ' + name + '!'


if __name__ == '__main__':
    WSGIServer(('172.4.2.20', 8880), app).serve_forever()
