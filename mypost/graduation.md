# 程序员成人礼

## 编程基础

- 实现数字
    - 数字的二进制表示及应用
    - 只用0和1的list（bitmap）表示数字，或者直接list长度表示数字，百年语言
    - linux 权限表示
    - modbuf 协议解析里的CRC，ID 卡通信协议的BCC，其它校验函数 checksum 等
    - 编程珠玑 bitmap 统计号码
    - 数据库 bitmap join index 提高性能
    - redis 内部的压缩数字 减少空间使用
    - 子网掩码
    - tcp 协议标志位 减少空间占用
    - Binary Indexed Tree
    - 各种语言里的 bitmap, bitset
- 实现字符串
- 实现时间
- 实现列表
- 实现字典
- 循环样式程序
- 函数式编程
- 二分查找
- 快速排序

### 数字

How do you set, clear, and toggle a single bit?
https://stackoverflow.com/questions/47981/how-do-you-set-clear-and-toggle-a-single-bit
Bit Twiddling Hacks
http://graphics.stanford.edu/~seander/bithacks.html
BitSet的用法
http://www.cnblogs.com/happyPawpaw/p/3823277.html
BitMap
https://blog.csdn.net/kl1106/article/details/79478787
Linux权限详解（chmod、600、644、666、700、711、755、777、4755、6755、7755）
https://blog.csdn.net/u013197629/article/details/73608613
Modbus通讯协议学习 - 认识篇
http://www.cnblogs.com/luomingui/archive/2013/06/14/Modbus.html
BCC(异或校验)、CRC、LRC校验算法
https://blog.csdn.net/windows19790408/article/details/5787039
位图（BitMap）索引
http://www.cnblogs.com/LBSer/p/3322630.html
关于 Bitmap join index
http://www.sizhxy.com/m/NewsCenter/NewsDetail-278.html
PHP实现 bitmap 位图排序 求交集
https://www.cnblogs.com/iLoveMyD/p/4167623.html
MySQL数据结构分析—BITMAP
http://blog.chinaunix.net/uid-26896862-id-3443593.html
ip地址及子网掩码换算，子网划分教程
https://jingyan.baidu.com/article/ae97a646d936ddbbfd461d02.html
Redis底层数据结构
https://zhuanlan.zhihu.com/p/38380467
Redis 设计与实现¶
http://redisbook.com/index.html

## 字符串

［Redis源码阅读］sds字符串实现
https://www.hoohack.me/2017/11/13/read-redis-src-sds
JavaScript字符串底层是如何实现的？
https://www.zhihu.com/question/51132164
Lua中字符串的实现
https://zhuanlan.zhihu.com/p/30757189?from_voters_page=true
C 字符串
http://www.runoob.com/cprogramming/c-strings.html
C语言字符串操作总结大全(超详细)
https://www.cnblogs.com/lidabo/p/5225868.html


转换成八进制数，则为 r=4, w=2, x=1, -=0（这也就是用数字设置权限时为何是4代表读，2代表写，1代表执行）

实际上，我们可以将所有的权限用二进制形式表现出来，并进一步转变成八进制数字：

rwx = 111 = 7
rw- = 110 = 6
r-x = 101 = 5
r-- = 100 = 4
-wx = 011 = 3
-w- = 010 = 2
--x = 001 = 1



## 网络基础

- tcp echo server
- web server
- 多路复用模型
- 多线程，同步，锁
- 日志
- 无锁队列
- 内存池，零拷贝实现
- 多进程，进程通信
- 信号处理，优雅重启

## 编译解析
- 状态机，url 解析，
- 配置解析，ini，xml，json
- 抽象语法树
- 正则引擎
- http 协议解析
- 实现一个模版
- 实现一个脚本

## db

- db 实现
- 索引实现
- 二分查找
- 树查找
- SQL 解析

## MVC

- Route
- Model
- View
- Controller
- Library
- Helper


## 

- udp 模拟 tcp
- 内存分配
- 多线程任务
- mapreduce