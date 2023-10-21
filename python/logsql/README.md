原始日志

    $ cat data.log
    10:11 111 222
    10:12 333 444
    10:14 333 666
    10:14 555 666

按规则解析日志，并只显示 time 列和 c 列

    $ python3 logsql.py data.log 'time b c' --select 'time c'
    10:11 222
    10:12 444
    10:14 666
    10:14 666

增加 where 条件过滤数据

    $ python3 logsql.py data.log 'time b c' --select 'time c' --where 'time=10:14'
    10:14 666
    10:14 666

分组统计

    $ python3 logsql.py data.log 'time b c' --select 'time count(*)' --group 'time'
    10:11 1
    10:12 1
    10:14 2
