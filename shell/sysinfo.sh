#!/bin/sh
#requires the following
# free, hostname, grep, cut, awk, uname
# http://www.dslreports.com/forum/remark,2069987

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
BOOT=`procinfo | grep Bootup | sed 's/Bootup: //g' | cut -f1-6 -d' '`
UPTIME=`uptime | cut -f5-8 -d' '`

PCIINFO=`lspci | cut -f3 -d':'`
#Another way to do it
#PCIINFO=`lspci | cut -f3 -d':'`

#print it out
echo "$HOSTNAME"
echo "----------------------------------"
echo "Hostname         : $HOSTNAME"
echo "Host Address(es) : $IP_ADDRS"
echo "Main Memory      : $MEMORY"
echo "Number of CPUs   : $CPUS"
echo "CPU Type         : $CPU_TYPE $CPU_TYPE2 $CPU_MHZ MHz"
echo "OS Name          : $OS_NAME"
echo "Kernel Version   : $OS_KERNEL"
echo "Bootup           : $BOOT - Uptime $UPTIME"
echo
echo "Devices"
echo "----------------------------------"
echo "$PCIINFO"
