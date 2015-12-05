# Redis复制原理

## 复制的目的

- 无状态服务：nginx, php
    - 没啥可复制的
    - 长连接的接入层服务，TCP连接状态也是状态
- 有状态服务：redis, mysql
    - 性能：读写分离
    - 可靠性, 挂了一个还有另一个

## 设计目标

- 不能丢数据，数据不能错
- 实时性
- 网络资源占用少

## 使用

    12345> slaveof 127.0.0.1 6379
    6379> set msg hello
    12345> get msg
    hello
    6379> del msg
    12345> exists msg
    0

## 旧版复制功能 2.8以前

- 同步:sync
    - 开始bgsave -> rdb
    - 开一个缓冲区记录之后的写操作 
    - bgsave完毕后，send给从，从load
    - 发送缓冲区命令给从
- 命令传播:command propagate
    - 主收到写命令，转发给从

## 旧版缺陷

- 初次复制，没问题
- 断线重连：全量复制
    - bgsave大量写，CPU,内存，磁盘IO消耗都很大
    - 发送rdb，占用大量带宽和流量
    = 从load rdb期间，无法处理请求


## 新版复制实现
- 断线重连：条件允许，会增量同步

- 主复制偏移量和从复制偏移量
    - 确定同步进度，一致性
- 主复制积压缓冲区
    - 固定长队列
    - 重连后增量同步
    - 默认1M：断线秒数*每秒写入量
- 服务器运行ID
    - 防止复制错服务器 

## psync命令实现

- psync <runid> <offset>
    - +fullresync <runid> <offset> : 全量
    - +continue: 主向从发增量
    - -err: 主版本低，客户端重新发起sync全量 

## 复制的实现

- 设置主服务ip,port: slaveof
- 建立socket连接
- 发送ping: 确认主状态正常
    - 超时，busy, pong
- 身份验证: requirepass, masterauth, auth
- 发送端口信息:replconf listening-port 12345
- 同步：全量或增量
- 命令传播

## 心跳检测

- 每秒一次：replconf ack <replication_offset>
- 目的：
    - 主从服务器的网络连接状态
        - info replication
        - slave0: ip, port, state, offset, lag
    - 辅助实现min-slaves
        - min-slaves-to-write 3
        - min-slaves-max-lag 10
    - 检测命令丢失
        - 检测offset发现命令传播丢失，进行补发 
