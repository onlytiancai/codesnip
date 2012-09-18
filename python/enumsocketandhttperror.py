import socket
import os

print "*" * 10 , 'all socket errorcode'
for code in socket.errno.errorcode:
    print "%s\t%s" % (code, os.strerror(code))

import BaseHTTPServer
codetable = BaseHTTPServer.BaseHTTPRequestHandler.responses
print "*" * 10 , 'all http errorcode'
for code in codetable:
    print "%s\t%s\t%s" % (code, codetable[code][0], codetable[code][1])
