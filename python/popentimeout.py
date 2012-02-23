#encoding=utf-8
'''
在python里调用子进程要防止子进程长时间运行,
如果处理一个web请求的时候需要调用子进程取结果,
一般可以超过一定时间后直接kill掉子进程

Popen一般会很快，不会阻塞太久，

但子进程的stdout.read方法是同步且不可以设置超时的
这样用gevent spawn一个协程开子进程read的话，
read会卡住整个协程，让所有的协程无法运行
而且用gevent.Timeout去设置超时也不管用

这就需要用到gevent.socket.wait_read了，
该方法可以等待一个文件描述符可读，
且等待可以设置超时参数，或用gevent.Timeout,
greenlet.get(timeout)来设置超时

refs:
https://bitbucket.org/denis/gevent/src/tip/examples/geventsendfile.py
'''
from subprocess import Popen, PIPE
from gevent.socket import wait_read
from sys import exc_info
try:
    from errno import EBADF
except ImportError:
    EBADF = 9

def runwithtimeout(cmd, timeout):
    process = Popen(cmd,shell=True, stdout=PIPE, stderr=PIPE,
            close_fds=True)
    stdout, stderr = None, None
    try:
        stdout = read(process.stdout, timeout)
        stderr = read(process.stderr, timeout)
    except Exception, ex:
        print 'popen timeout:', cmd
    finally:
        process.kill()
        return stdout, stderr

def read(fromfile, timeout):
    while True:
        try:
            wait_read(fromfile.fileno(), timeout)
            return fromfile.read()
        except:
            ex = exc_info()[1]
            if ex.args[0] == EBADF:
                return ''
            raise
   
if __name__ == '__main__':
    from gevent import spawn
    from traceback import format_exception
    cmd = 'curl -I --connect-timeout %s -m %s %s ' \
            % (5, 5, '202.99.160.68')
    task = spawn(runwithtimeout, cmd, 10)
    try:
        print ''.join(task.get())
    except:
        type, value, tb = exc_info()
        print ''.join(format_exception(type,value, tb))
