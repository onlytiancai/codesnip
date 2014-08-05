# -*- encoding: utf-8 -*-
'''
== 简介

这是一个分布式任务处理框架，适用于批量的任务处理，提高速度,
任务会下发给多个进程或多台机器去分布式的处理。

== 基本使用

生产任务：往全局q里压任务就行
客户端：在process_request回调里执行任务
服务端：在process_result回调里处理结果

== 原理

Server起来后会监听一个端口，多个Client连接到Server，拉取自己的任务，
Client处理完任务，无论成功还是失败，都会返回给Server，由Server去做后续工作。

Server会为每一个Client开一个线程去收发包，处理结果，所以有几十个连接还是可以的，再
多就够呛了。

Server和Client的启动顺序无所谓，任何一个挂掉后重启，整个系统会自动继续运行。

服务端任务的产生依靠一个全局的Queue来进行数据交换，所以要有个生产者线程来不断的往
这个队列里压任务，该队列是阻塞队列，队列满时会自动阻塞生产者线程, 具体可参考示例里的
Producer类。

服务端还需要处理各个Client的任务执行结果, 需要继承Connection类，并重写
process_result的模板方法, 无论任务处理成功还是失败，都会调用该方法，任务的原始请求，
错误信息和处理结果都可以在参数里取到，具体参考示例的MyConn类。

客户端负责具体执行任务，需要继承Client类，并重写process_request模板方法，
该方法可以返回的结果和抛出的异常会传递给Server。

最后服务端要把生产者线程启动起来，然后把server启动起来，最后主线程可以阻塞在
server的wait()方法上。客户端的话直接调用run就可以了，会自动连接到server拉任务。

客户端执行任务是串行的，因为要保持封包拆包简单一些，否则每个包得有需要，服务端收到
后还得和请求做匹配才能执行用户回调，比较麻烦。所以要想提高并行度，多开几个Client就
行了。

== 服务端Todo

- done: 踢掉空闲连接
    - 没设计应用层心跳，如果recv超过1分钟，直接关闭连接，等待重连
- done: 任务执行超时的处理策略
    - 如果某个Client执行任务超时，表现为recv超时，直接关闭连接, 等待重连
- done: 任务执行出错的处理策略
    - Client给Server的应答包第一行为状态行，ok表成功，其它表执行出错，传递给应用层
- done: 应用处理结果时，能拿到对应的请求
    - 已完成，这样Connection处理结果时可拿到上下文
- done: 取数据时对端断开，要把已读取的数据传给应用
    - 分两种情况，一个是读完结果后断开，一个是没读完整备断开，
    无论哪种，都要执行用户设置的回调，否则就丢任务了
- todo: 过载保护，超过100连接后直接拒绝新连接
- todo: Server挂掉的话，队列里的任务如何恢复

== 客户端Todo

- done: 客户端抛出异常后传递给服务端
- done: 断线重连

'''
import logging
import Queue
import threading
import socket
import select
import time

end_flag = '\r\n\r\n\r\n\r\n'
q = Queue.Queue(10)


class Connection(threading.Thread):
    '表示一条客户端连接'
    recv_timeout = 60

    def __init__(self, socket):
        super(Connection, self).__init__()
        self.setDaemon(True)

        self.socket = socket
        self.socket.setblocking(0)
        self.peer = self.socket.getpeername()
        self.connected = True
        self.error = None

    def _unpack(self, buf):
        '''
        拆包，第一行表示状态，ok表示处理成功，其它表示出错, 第一行之后的为内容
        '''
        if self.error:
            return self.error, buf
        pos = buf.find('\n')
        if pos == -1:
            self.error = 'get first line error'
        error = buf[:pos].strip()
        if error == 'ok':
            error = None
        body = buf[pos + 1:len(buf) - len(end_flag)]
        return error, body

    def _recvmsg(self):
        '返回错误原因和结果'
        buff = b""

        while True:
            ready = select.select([self.socket], [], [], self.recv_timeout)
            if ready[0]:
                data = self.socket.recv(1024)
                logging.debug('conn recv:%s %s', self.peer, repr(data))
                if len(data) == 0:
                    self.connected = False
                    break
                buff += data
                if buff.endswith(end_flag):
                    break
            else:
                self.error = 'recv timeout'
                self.socket.shutdown(socket.SHUT_RDWR)
                break
        return self._unpack(buff)

    def run(self):
        '从队列里获取任务，发给Client，收集结果，最后执行用户回调'
        logging.info("conn thread started:%s", self.peer)
        while True:
            try:
                item = q.get()
                self.socket.sendall('%s%s' % (item, end_flag))
                error, body = self._recvmsg()
                if self.error:
                    error = self.error

                logging.debug("conn recv msg:%s %s", self.peer, repr(body))
                self.process_result(item, error, body)

                if not self.connected:
                    logging.info('client closed:%s', self.peer)
                    break

                if self.error:
                    logging.info('client error:%s', self.error)
                    break

            except Exception, ex:
                message = ex.strerror if hasattr(ex, 'strerror') else ex.message
                logging.info("conn error:%s %s", self.peer, message)
                break

    def process_result(self, req, error, result):
        '''
        子类实现该方法，处理结果
        req表示原始的任务，error不为空表示任务处理出错，result是处理结果
        '''
        pass


class Server(threading.Thread):
    '监听端口，为每个Client连接分配一个Connection对象并启动一个线程'
    def __init__(self, listenep, ConnClass):
        super(Server, self).__init__()
        self.setDaemon(True)

        self.listenep = listenep
        self.ConnClass = ConnClass
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind(self.listenep)
        self.socket.listen(1024)

        self.cond = threading.Condition()
        self.cond.acquire()

    def run(self):
        logging.info("server is runing:%s", self.listenep)
        while True:
            client, addr = self.socket.accept()
            logging.info('new conn:%s', addr)
            conn = self.ConnClass(client)
            conn.start()
        self.cond.release()

    def wait(self):
        while True:
            self.cond.wait(0.1)


class Client(object):
    '客户端类，自动连到Server拉任务，执行出错或成功后返回给Server'
    def __init__(self, addr):
        self.addr = addr
        self.client = None

    def _connect(self):
        '不断连Server直到连上'
        while True:
            try:
                self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.client.connect(self.addr)
                logging.info("connect %s ok", self.addr)
                break
            except socket.error, ex:
                time.sleep(1)
                logging.info("connect %s faild:%s", self.addr, ex.strerror)

    def _recvtask(self):
        '读取一个任务'
        ret = b''
        while True:
            data = self.client.recv(1024)
            if not data:
                logging.info("conn closed")
                return 'conn closed', ret
            ret += data
            if ret.endswith(end_flag):
                return None, ret[:len(ret) - len(end_flag)]

    def run(self):
        '读取任务，执行任务，返回执行结果'
        self._connect()
        while True:
            try:
                error, request = self._recvtask()
                logging.info("recv data:%s", repr(request))

                result = ''
                status = 'ok'
                try:
                    result = self.process_request(request)
                except Exception, ex:
                    status = ex.message.replace('\n', '')
                result = '%s\n%s%s' % (status, result, end_flag)
                self.client.sendall(result)

                if error:
                    logging.info('read line error:%s', error)
                    self._connect()
            except Exception, ex:
                message = ex.strerror if hasattr(ex, 'strerror') else ex.message
                logging.info("client run error:%s", message)
                self._connect()

    def process_request(self, req):
        '需要子类实现, 处理Client需要完成的任务, 可抛出异常'
        logging.info("Client Default process_request:%s", req)
        time.sleep(1)


# 以下为测试及示例代码, 演示一个分布式的字符串反转任务处理
class MyConn(Connection):
    def __init__(self, socket):
        super(MyConn, self).__init__(socket)

    def process_result(self, request, error, result):
        logging.info("process_result:%s %s %s", request, error, repr(result))


class MyClient(Client):
    def __init__(self, addr):
        super(MyClient, self).__init__(addr)

    def process_request(self, req):
        time.sleep(1)
        return req[::-1]


class Producer(threading.Thread):
    def __init__(self):
        super(Producer, self).__init__()
        self.setDaemon(True)

    def run(self):
        logging.info("producer thread runing")
        i = 0
        while True:
            q.put('%s.com' % i)
            logging.info("producer put %s", i)
            i += 1


if __name__ == '__main__':
    # python distributed_task.py 启动测试Server
    # python distributed_task.py client 启动测试Client，可启多个
    import sys
    logging.basicConfig(level=logging.INFO)

    server_addr = ('172.4.2.20', 8002)
    if len(sys.argv) == 1:
        producer = Producer()
        producer.start()

        server = Server(server_addr, MyConn)
        server.start()
        server.wait()
    elif sys.argv[1] == 'client':
        client = MyClient(server_addr)
        client.run()
