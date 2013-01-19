### http hello world 性能测试大比评

目前参与测试的有如下

1. gevent.pywsgi
1. 使用了多核的gevent.pywsgi
1. gunicorn
1. nginx
1. node
1. golang
1. erlang
1. 我自己用http_parser和libuv写的http demo server

测试脚本

    ab -r -n  100000 -c 10 http://localhost:7000/


系统信息

    # cat /proc/cpuinfo | grep name | cut -f2 -d:|uniq -c
    2  Intel(R) Core(TM)2 Duo CPU     P8400  @ 2.26GHz

    # cat /proc/meminfo | grep MemTotal
    MemTotal:        1994472 kB

    $ uname -a
    Linux huhao-ThinkPad-X200 3.2.0-29-generic #46-Ubuntu SMP Fri Jul 27 17:03:23 UTC 2012 x86_64 x86_64 x86_64 GNU/Linux

    $ python -V
    Python 2.7.3

    $ pip freeze
    gevent==1.0dev
    greenlet==0.4.0
    gunicorn==0.17.2

    $ node -v
    v0.6.12

    $ nginx -v
    nginx version: nginx/1.1.19

    $go version
    go version go1

    $ erl -version
    Erlang (SMP,ASYNC_THREADS) (BEAM) emulator version 5.8.5


python gevent_webserver.py

    Concurrency Level:      10
    Time taken for tests:   26.945 seconds
    Complete requests:      100000
    Failed requests:        0
    Write errors:           0
    Total transferred:      13800000 bytes
    HTML transferred:       1800000 bytes
    Requests per second:    3711.31 [#/sec] (mean)
    Time per request:       2.694 [ms] (mean)
    Time per request:       0.269 [ms] (mean, across all concurrent requests)
    Transfer rate:          500.16 [Kbytes/sec] received

gunicorn -w `cat /proc/cpuinfo | grep processor | wc | awk '{print $1}'` gunicorn_app:app -b 0.0.0.0:7000

    Concurrency Level:      10
    Time taken for tests:   24.033 seconds
    Complete requests:      100000
    Failed requests:        0
    Write errors:           0
    Total transferred:      16000000 bytes
    HTML transferred:       1400000 bytes
    Requests per second:    4160.98 [#/sec] (mean)
    Time per request:       2.403 [ms] (mean)
    Time per request:       0.240 [ms] (mean, across all concurrent requests)
    Transfer rate:          650.15 [Kbytes/sec] received

python multi_cpu_gevent_webserver.py

    Concurrency Level:      10
    Time taken for tests:   19.148 seconds
    Complete requests:      100000
    Failed requests:        0
    Write errors:           0
    Total transferred:      13800000 bytes
    HTML transferred:       1800000 bytes
    Requests per second:    5222.50 [#/sec] (mean)
    Time per request:       1.915 [ms] (mean)
    Time per request:       0.191 [ms] (mean, across all concurrent requests)
    Transfer rate:          703.81 [Kbytes/sec] received

node ./node_app.js

    Concurrency Level:      10
    Time taken for tests:   13.976 seconds
    Complete requests:      100000
    Failed requests:        0
    Write errors:           0
    Total transferred:      7600000 bytes
    HTML transferred:       1200000 bytes
    Requests per second:    7155.30 [#/sec] (mean)
    Time per request:       1.398 [ms] (mean)
    Time per request:       0.140 [ms] (mean, across all concurrent requests)
    Transfer rate:          531.06 [Kbytes/sec] received

go run go_app.go

    Concurrency Level:      10
    Time taken for tests:   13.582 seconds
    Complete requests:      100000
    Failed requests:        0
    Write errors:           0
    Total transferred:      10900000 bytes
    HTML transferred:       1200000 bytes
    Requests per second:    7362.84 [#/sec] (mean)
    Time per request:       1.358 [ms] (mean)
    Time per request:       0.136 [ms] (mean, across all concurrent requests)
    Transfer rate:          783.74 [Kbytes/sec] received



sudo nginx -p `pwd`/ -c nginx_app.conf

    Concurrency Level:      10
    Time taken for tests:   7.475 seconds
    Complete requests:      100000
    Failed requests:        0
    Write errors:           0
    Total transferred:      17100000 bytes
    HTML transferred:       1400000 bytes
    Requests per second:    13377.66 [#/sec] (mean)
    Time per request:       0.748 [ms] (mean)
    Time per request:       0.075 [ms] (mean, across all concurrent requests)
    Transfer rate:          2233.96 [Kbytes/sec] received

./run3 //我自己写的http server, 单线程

    Concurrency Level:      10
    Time taken for tests:   7.417 seconds
    Complete requests:      100000
    Failed requests:        0
    Write errors:           0
    Total transferred:      7900000 bytes
    HTML transferred:       1400000 bytes
    Requests per second:    13483.31 [#/sec] (mean)
    Time per request:       0.742 [ms] (mean)
    Time per request:       0.074 [ms] (mean, across all concurrent requests)
    Transfer rate:          1040.22 [Kbytes/sec] received

erlc er_app.erl
erl -noshell -s er_app start


