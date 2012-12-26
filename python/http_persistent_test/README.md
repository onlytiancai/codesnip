## 测试支持HTTP持久连接的客户端和服务端

### 所需模块

1. 支持http persistent connection的客户端组件选用urllib3, 或requests
1. 支持http persistent connection的服务端组件选用gunicor
1. 其余还需要web.py和gevent

### 准备工作

1. 新建一个virtual_env环境
1. 加载这个环境
1. pip install -r requirements.txt
1. 需要两台机器，一台跑服务端(这里是172.4.2.20)，一台跑客户端

### 启动服务端

如下

    sh ./start_server.sh

### 启动客户端

    python client_with_urllib3.py
    python client_with_requests.py 

### 检测连接情况

预期只建立10个TCP连接，并且处理完所有请求，只有10个ESTABLISHED状态的连接

    netstat -tnp | grep 8880

结果符合预期

    (Not all processes could be identified, non-owned process info
      will not be shown, you would have to be root to see it all.)
    tcp        0      0 192.168.1.119:39788     172.4.2.20:8880         ESTABLISHED 1844/python     
    tcp        0      0 192.168.1.119:39786     172.4.2.20:8880         TIME_WAIT   -               
    tcp        0      0 192.168.1.119:39791     172.4.2.20:8880         ESTABLISHED 1844/python     
    tcp        0      0 192.168.1.119:39790     172.4.2.20:8880         ESTABLISHED 1844/python     
    tcp        0      0 192.168.1.119:39756     172.4.2.20:8880         ESTABLISHED 1844/python     
    tcp        0      0 192.168.1.119:39784     172.4.2.20:8880         ESTABLISHED 1844/python     
    tcp        0      0 192.168.1.119:39782     172.4.2.20:8880         TIME_WAIT   -               
    tcp        0      0 192.168.1.119:39753     172.4.2.20:8880         TIME_WAIT   -               
    tcp        0      0 192.168.1.119:39789     172.4.2.20:8880         ESTABLISHED 1844/python     
    tcp        0      0 192.168.1.119:39785     172.4.2.20:8880         ESTABLISHED 1844/python     
    tcp        0      0 192.168.1.119:39793     172.4.2.20:8880         ESTABLISHED 1844/python     
    tcp        0      0 192.168.1.119:39759     172.4.2.20:8880         ESTABLISHED 1844/python     
    tcp        0      0 192.168.1.119:39749     172.4.2.20:8880         ESTABLISHED 1844/python 

抓包查看http协议里是否声明了keep-alive

    sudo tcpdump -A -s0 -i eth0 port 8880

结果符合预期

    GET /2378 HTTP/1.1
    Host: 172.4.2.20:8880
    Accept-Encoding: identity


    HTTP/1.1 200 OK
    Server: gunicorn/0.17.0
    Date: Wed, 26 Dec 2012 05:39:29 GMT
    Connection: keep-alive
    Transfer-Encoding: chunked

    Hello, 2369!
