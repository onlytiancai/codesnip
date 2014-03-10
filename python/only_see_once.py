# -*- coding: utf-8 -*-

import web
import uuid
import shelve

tpl_index = '''
<html>
<head>
<meta charset="UTF-8" />
</head>
<body>
    <h1>阅后即焚</h2>
    <h3>写一段话，生成一个网页，只有第一个人可以看到</h3>
    <form action="/gen" method="post" accept-charset="utf-8">
        <p><input type="text" name="content" value="" placeholder="请输入你想说的话"></p>
        <p><input type="submit" value="生成网页"></p>
    </form>
</body>
</html>
'''

tpl_gen = '''
<p>请复制如下网址发送给你的好友，你不要打开哦。</p>
<p>http://172.4.2.20:8805/show/%s</p>
'''


class index(object):
    def GET(self):
        web.header('Content-Type', 'text/html; charset=utf-8', unique=True)
        return tpl_index 


class gen(object):
    def POST(self):
        try:
            db = shelve.open('only_see_once.db')
            content = web.input().content
            content_id = str(uuid.uuid1()).encode('ascii')
            db['content.' + content_id] = content
            return tpl_gen % content_id
        finally:
            db.close()


class show(object):
    def GET(self, content_id):
        try:
            web.header('Content-Type', 'text/html; charset=utf-8', unique=True)
            content_id = content_id.decode('utf-8').encode('ascii')
            db = shelve.open('only_see_once.db')
            if 'content.' + content_id not in db:
                return web.notfound()

            result = u"Sorry, 你无权访问"
            content = db.get('content.' + content_id, 'miss u.')
            if 'cookie.' + content_id not in db:
                cookie_id = str(uuid.uuid1()).encode('ascii')
                db['cookie.' + content_id] = cookie_id
                web.setcookie('cookie.' + content_id, cookie_id)
                result = content
            else:
                cookie_id = db['cookie.' + content_id]

            cookies = web.cookies()
            client_cookie_id = cookies.get('cookie.' + content_id, '').encode('ascii')
            if cookie_id == client_cookie_id:
                result = content

            return result

        finally:
            db.close()


urls = ['/', index,
        '/show/([^/]*)', show,
        '/gen', gen,
        ]

app = web.application(urls, globals())
wsgiapp = app.wsgifunc()

if __name__ == '__main__':
    app.run()
