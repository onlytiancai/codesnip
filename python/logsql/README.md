## 项目介绍

用 SQL 直接分析任何文本日志，不需要 awk 复杂的语法，也不需要把日志写入 DB 或 Elasticsearch。

## Quick Start 

原始日志

    $ cat data.log
    10:11 111 222
    10:12 333 444
    10:14 333 666
    10:14 555 666

按规则解析日志，并只显示 time 列和 c 列

    $ python3 logsql.py data.log 'time b c' --select 'time,c' 
    {'time': '10:11'}
    {'time': '10:12'}
    {'time': '10:14'}

增加 where 条件过滤数据

    $ logsql.py data.log 'time b c' --select 'time,c' --where 'time=="10:14"' 
    {'time': '10:14', 'c': '666'}
    {'time': '10:14', 'c': '666'}

分组统计

    $ python3 logsql.py data.log 'time b c' --select 'time,avg(c)' --group 'time' 
    {'avg(c)': 222.0, 'time': '10:11'}
    {'avg(c)': 444.0, 'time': '10:12'}
    {'avg(c)': 666.0, 'time': '10:14'}

