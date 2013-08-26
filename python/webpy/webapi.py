from utils import safestr
from threading import local as threadlocal

class HTTPError(Exception):
    def __init__(self, status, headers={'Content-Type': 'text/html'}, data=""):
        ctx.status = status
        for k, v in headers.items():
            header(k, v)
        self.data = data
        Exception.__init__(self, status)
 
ctx = context = threadlocal()

def header(hdr, value, unique=False):
    hdr, value = safestr(hdr), safestr(value)
    if '\n' in hdr or '\r' in hdr or '\n' in value or '\r' in value:
        raise ValueError('invalid characters in header')
        
    if unique is True:
        for h, v in ctx.headers:
            if h.lower() == hdr.lower():
                return
    
    ctx.headers.append((hdr, value))
