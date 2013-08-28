'''测试脚本：curl -v localhost:8000/aaaa?name=bbb -d "name=ccc" -H "Cookie: name=ddd"'''

import logging
import web

logging.basicConfig(level=logging.NOTSET, format='%(asctime)s %(levelname)s %(module)s %(message)s')
        
urls = (
    '/(.*)', 'hello'
)

app = web.application(urls, globals())

class hello:        
    def GET(self, name):
        web.setcookie('name', name)
        yield 'Hello, ' + name + '!\n'
        yield 'input.name = %s\n' % web.input()["name"]
        yield 'cookie.name = %s\n' % web.cookies()["name"].value

    POST = GET

if __name__ == "__main__":
    app.run()
