import web
import urllib2
import urllib
import json

class OAuth(object):
    def __init__(self, name, client_id, client_secret, base_url, access_token_url, authorize_url):
        self.name = name
        self.client_id = client_id
        self.client_secret = client_secret
        self.base_url = base_url
        self.access_token_url = access_token_url
        self.authorize_url = authorize_url

    def _request(self, method, uri, data):
        if method == 'GET':
            uri = uri + '?'  + urllib.urlencode(data)
            print uri
            response = urllib2.urlopen(uri)
        else:
            response = urllib2.urlopen(uri, urllib.urlencode(data))
        return json.loads(response.read())

    def get_authorize_url(self, **kargs):
        data = dict(client_id=self.client_id) 
        data.update(kargs)
        return self.authorize_url + '?'  + urllib.urlencode(data) 

    def get_access_token(self, method, **kargs):
        data = dict(client_id=self.client_id, client_secret=self.client_secret) 
        data.update(kargs)
        return self._request(method=method, uri=self.access_token_url, data=data)

    def request(self, method,  uri, **kargs):
        if not uri.startswith('http://'):
            uri = self.base_url + uri
        return self._request(method=method, uri=uri, data=kargs)


service = OAuth(
           name='dnspod',
           client_id='10003',
           client_secret='nicai',
           base_url='https://www.dnspod.cn/api/',
           access_token_url='https://www.dnspod.cn/oauth/access.token',
           authorize_url='https://www.dnspod.cn/oauth/authorize')

class index(object):
    def GET(self):
        access_token = web.cookies().get('access_token')
        if not access_token:
            return '<a href="/login">login</a>'
        data = dict(format='json', lang='en',error_on_empty='no', access_token=access_token)
        result = service.request('POST', 'Domain.List', **data)
        return '<a href="/logout">logout</a></br><pre>%s</pre>' % json.dumps(result, indent=4)

class login(object):
    def GET(self):
        url = service.get_authorize_url(response_type='code', redirect_uri='http://www.abc.com/callback', state='abc')
        return web.redirect(url)

class logout(object):
    def GET(self):
        web.setcookie('access_token', None, -1)
        return web.redirect('/')

class callback(object):
    def GET(self):
        code = web.input().code
        if code:
            result = service.get_access_token('GET', code=code, grant_type='authorization_code', redirect_uri='http://www.abc.com/callback')
            access_token = result['access_token']
            web.setcookie('access_token', access_token)
            return web.redirect('/')

urls = ("/"          , 'index',
       "/login"      , 'login',
       "/logout"      , 'logout',
       "/callback"   , 'callback',
)

web.config.debug = True
app = web.application(urls, globals())
if __name__ == '__main__':
    app.run()
