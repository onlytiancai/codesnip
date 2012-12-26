from gevent import monkey
monkey.patch_all()
import gevent
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
