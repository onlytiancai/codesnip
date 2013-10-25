#!/bin/sh

HOSTNAME=`hostname -s`
IP_ADDRS=`ifconfig | grep 'inet addr' | grep -v '255.0.0.0' | cut -f2 -d':' | awk '{print $1}'`
IP_ADDRS=`echo $IP_ADDRS | sed 's/\n//g'`

#memory
MEMORY=`free | grep Mem | awk '{print $2}'`

#cpu info
CPUS=`cat /proc/cpuinfo | grep processor | wc -l | awk '{print $1}'`
CPU_MHZ=`cat /proc/cpuinfo | grep MHz | tail -n1 | awk '{print $4}'`
CPU_TYPE=`cat /proc/cpuinfo | grep vendor_id | tail -n 1 | awk '{print $3}'`
CPU_TYPE2=`uname -m`

OS_NAME=`uname -s`
OS_KERNEL=`uname -r`
UPTIME=`uptime`
PROC_COUNT=`ps -ef | wc -l`

body() {
    IFS= read -r header
    printf '%s\n' "$header"
    "$@"
}

#print it out
echo "概要信息"
echo "----------------------------------"
echo "主机名            : $HOSTNAME"
echo "IP地址            : $IP_ADDRS"
echo "内存大小          : $MEMORY"
echo "CPU核数           : $CPUS"
echo "CPU类型           : $CPU_TYPE $CPU_TYPE2 $CPU_MHZ MHz"
echo "操作系统          : $OS_NAME"
echo "内核版本          : $OS_KERNEL"
echo "进程总数          : $PROC_COUNT"
echo "启动时间及负载    : $UPTIME"
echo
echo "内存使用情况"
echo "----------------------------------"
free -m
echo 
echo "磁盘使用情况"
echo "----------------------------------"
df -h
echo 
echo "网络连接情况"
echo "----------------------------------"
netstat -n | awk '/^tcp/ {++S[$NF]} END {for(a in S) print a, S[a]}'
echo 
echo "网络监听情况"
echo "----------------------------------"
netstat -tnpl | awk 'NR>2 {printf "%-20s %-15s \n",$4,$7}'
echo 
echo "内存占用Top 10"
echo "----------------------------------"
ps -eo rss,pmem,pcpu,vsize,args |body sort -k 1 -r -n | head -n 10
echo 
echo "CPU占用Top 10"
echo "----------------------------------"
ps -eo rss,pmem,pcpu,vsize,args |body sort -k 3 -r -n | head -n 10
echo 
echo "最近1小时网络流量统计"
echo "----------------------------------"
sar -n DEV -s `date -d "1 hour ago" +%H:%M:%S`
echo 
echo "最近1小时cpu使用统计"
echo "----------------------------------"
sar -u -s `date -d "1 hour ago" +%H:%M:%S`
echo 
echo "最近1小时磁盘IO统计"
echo "----------------------------------"
sar -b -s `date -d "1 hour ago" +%H:%M:%S`
echo 
echo "最近1小时进程队列和平均负载统计"
echo "----------------------------------"
sar -q -s `date -d "1 hour ago" +%H:%M:%S`
echo 
echo "最近1小时内存和交换空间的统计统计"
echo "----------------------------------"
sar -r -s `date -d "1 hour ago" +%H:%M:%S`
echo 

# 参考链接：
# http://www.dslreports.com/forum/remark,2069987
# http://www.ctohome.com/FuWuQi/1b/688.html
