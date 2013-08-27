import logging
import application

logging.basicConfig(level=logging.NOTSET, format='%(asctime)s %(levelname)s %(module)s %(message)s')
        
urls = (
    '/(.*)', 'hello'
)

app = application.application(urls, globals())

class hello:        
    def GET(self, name):
        if not name: 
            name = 'World'
        return 'Hello, ' + name + '!\n'

if __name__ == "__main__":
    app.run()
