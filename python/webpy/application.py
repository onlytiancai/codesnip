from wsgiref.simple_server import make_server
import types

import webapi as web
import utils


class application(object):
    def __init__(self, mapping=(), fvars={}):
        self.init_mapping(mapping)
        self.fvars = fvars

    def init_mapping(self, mapping):
        self.mapping = list(utils.group(mapping, 2))

    def wsgifunc(self, *middleware):
        def mywsgi(env, start_resp):
            self.load(env)
            result = self.handle()

            status, headers = web.ctx.status, web.ctx.headers
            start_resp(status, headers)
            result = web.safestr(iter(result))
            return result

        for m in middleware:
            mywsgi = m(mywsgi)

        return mywsgi

    def handle(self):
        try:
            host = web.ctx.host.split(':')[0]  # strip port
            fn, args = self._match(self.mapping, host)
            return self._delegate(fn, self.fvars, args)
        except web.HTTPError, ex:
            return ex.message 
 
    def _delegate(self, f, fvars, args=[]):
        def handle_class(cls):
            meth = web.ctx.method
            if not hasattr(cls, meth):
                raise web.nomethod(cls)
            tocall = getattr(cls(), meth)
            return tocall(*args)

        def is_class(o):
            return isinstance(o, (types.ClassType, type))

        if f is None:
            raise web.HTTPError('404 Not Found') 
        elif is_class(f):
            return handle_class(f)
        else:
            raise web.HTTPError('404 Not Found') 

    def _match(self, mapping, value):
        for pat, what in mapping:
            if isinstance(what, basestring):
                what, result = utils.re_subm('^' + pat + '$', what, value)
            else:
                result = utils.re_compile('^' + pat + '$').match(value)
                
            if result:  # it's a match
                return what, [x for x in result.groups()]
        return None, None
 
    def load(self, env):
        """Initializes ctx using env."""
        ctx = web.ctx
        ctx.status = '200 OK'
        ctx.headers = []
        ctx.output = ''
        ctx.environ = ctx.env = env
        ctx.host = env.get('HTTP_HOST')
        ctx.ip = env.get('REMOTE_ADDR')
        ctx.method = env.get('REQUEST_METHOD')
        ctx.path = env.get('PATH_INFO')

    def run(self, *middleware):
        import sys
        ip = sys.argv[1] if len(sys.argv) > 1 else 'localhost'
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
        httpd = make_server(ip, port, self.wsgifunc(*middleware))
        httpd.serve_forever()
