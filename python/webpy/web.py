from wsgiref.simple_server import make_server
from threading import local as threadlocal
import os, cgi, types, logging, itertools, re, Cookie, urllib # NOQA

ctx = context = threadlocal()


class application(object):
    def __init__(self, mapping=(), fvars={}):
        self.mapping = [(mapping[i], mapping[i + 1]) for i in range(0, len(mapping), 2)]
        self.fvars = fvars

    def wsgifunc(self, *middleware):
        def mywsgi(env, start_resp):
            self.load(env)
            result = self.handle()

            status, headers = ctx.status, ctx.headers
            start_resp(status, headers)
            result = safestr(iter(result))
            return result

        for m in middleware:
            mywsgi = m(mywsgi)

        return mywsgi

    def handle(self):
        try:
            fn, args = self._match(self.mapping, ctx.path)
            logging.debug('handle:%s %s', fn, args)
            return self._delegate(fn, self.fvars, args)
        except HTTPError, ex:
            return ex.message
 
    def _delegate(self, f, fvars, args=[]):
        if f is None:
            raise HTTPError('404 Not Found')
        elif isinstance(f, (types.ClassType, type)):
            return handle_class(f, args)
        elif isinstance(f, basestring):
            if '.' in f:
                mod, cls = f.rsplit('.', 1)
                mod = __import__(mod, None, None, [''])
                cls = getattr(mod, cls)
            else:
                cls = fvars[f]
            return handle_class(cls, args)
        elif hasattr(f, '__call__'):
            return f()
        else:
            raise HTTPError('404 Not Found')

    def _match(self, mapping, value):
        for pat, what in mapping:
            if isinstance(what, basestring):
                what, result = re_subm('^' + pat + '$', what, value)
            else:
                result = re.compile('^' + pat + '$').match(value)
                
            logging.debug('_match: %s %s %s %s', pat, what, value, result)
            if result:
                return what, [x for x in result.groups()]
        return None, None
 
    def load(self, env):
        """Initializes ctx using env."""
        ctx.__dict__.clear()
        ctx.status = '200 OK'
        ctx.headers = []
        ctx.output = ''
        ctx.environ = ctx.env = env
        ctx.host = env.get('HTTP_HOST')
        ctx.ip = env.get('REMOTE_ADDR')
        ctx.method = env.get('REQUEST_METHOD')
        ctx.path = env.get('PATH_INFO')
        ctx.homepath = os.environ.get('REAL_SCRIPT_NAME', env.get('SCRIPT_NAME', ''))


    def run(self, *middleware):
        import sys
        ip = sys.argv[1] if len(sys.argv) > 1 else 'localhost'
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
        logging.info('app start with %s:%s', ip, port)
        httpd = make_server(ip, port, self.wsgifunc(*middleware))
        httpd.serve_forever()


def safestr(obj, encoding='utf-8'):
    if isinstance(obj, unicode):
        return obj.encode(encoding)
    elif isinstance(obj, str):
        return obj
    elif hasattr(obj, 'next'):  # iterator
        return itertools.imap(safestr, obj)
    else:
        return str(obj)


def handle_class(cls, args):
    meth = ctx.method
    if not hasattr(cls, meth):
        raise HTTPError('405 Method Not Allowed')
    tocall = getattr(cls(), meth)
    return tocall(*args)


def re_subm(pat, repl, string):
    class _re_subm_proxy:
        def __init__(self):
            self.match = None

        def __call__(self, match):
            self.match = match
            return ''

    compiled_pat = re.compile(pat)
    proxy = _re_subm_proxy()
    compiled_pat.sub(proxy, string)
    return compiled_pat.sub(repl, string), proxy.match


class HTTPError(Exception):
    def __init__(self, status, headers={'Content-Type': 'text/html'}, data=""):
        ctx.status = status
        for k, v in headers.items():
            header(k, v)
        self.data = data
        Exception.__init__(self, status)
 

def header(hdr, value, unique=False):
    hdr, value = safestr(hdr), safestr(value)
    if '\n' in hdr or '\r' in hdr or '\n' in value or '\r' in value:
        raise ValueError('invalid characters in header')
        
    if unique is True:
        for h, v in ctx.headers:
            if h.lower() == hdr.lower():
                return
    
    ctx.headers.append((hdr, value))


def setcookie(name, value, expires='', domain=None,
              secure=False, httponly=False, path=None):
    morsel = Cookie.Morsel()
    name, value = safestr(name), safestr(value)
    morsel.set(name, value, urllib.quote(value))
    if expires < 0:
        expires = -1000000000
    morsel['expires'] = expires
    morsel['path'] = path or ctx.homepath + '/'
    if domain:
        morsel['domain'] = domain
    if secure:
        morsel['secure'] = secure
    value = morsel.OutputString()
    if httponly:
        value += '; httponly'
    header('Set-Cookie', value)


def cookies():
    thiscookie = Cookie.SimpleCookie()
    if 'HTTP_COOKIE' in ctx.env:
        thiscookie.load(ctx.env['HTTP_COOKIE'])
        return thiscookie


def data():
    if 'data' not in ctx.__dict__:
        cl = ctx.env.get('CONTENT_LENGTH')
        cl = int(cl) if cl.isdigit() else 0
        ctx.data = ctx.env['wsgi.input'].read(cl)
    return ctx.data


def input():
    query_string= cgi.parse_qs(ctx.env['QUERY_STRING'])
    post_data = cgi.parse_qs(data())
    for key in post_data:
        if key in query_string:
            query_string[key].extend(post_data[key])
        else:
            query_string[key] = post_data
    return query_string 
