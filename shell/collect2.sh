# 脚本说明
#     本脚本用来收集本机的CPU，内存，磁盘等信息到graphite
#
# 如何使用：执行如下语句，可以加入到/etc/rc.local里设置开机自动启动
#     bash <(curl https://raw2.github.com/onlytiancai/codesnip/master/shell/collect2.sh -s) $API_KEY
#
# 如何关掉agent 
#     ps -ef | grep $API_KEY | grep -v grep | cut -c 9-15 | xargs kill -9


# 参数检查
if [ ! -n "$1" ] ;then
    echo "you have not input the API_KEY!"
    exit 1
fi

API_KEY=${API_KEY:=$1}
COLLECTOR_IP=${COLLECTOR_IP:="collector.monitor.dnspod.cn"}
COLLECTOR_PORT="2003"
DEBUG=1


# 如果进程存在，则先杀掉
PID=$$
ps -ef | grep $API_KEY | grep -v grep | grep -v " $PID " | cut -c 9-15 | xargs kill


# 获取IP地址，主机等信息
IP_ADDRS="`LC_ALL=en /sbin/ifconfig | grep 'inet addr' | grep -v '255.0.0.0' \
    | head -n1 | cut -f2 -d':' | awk '{print $1}'`"
if [ -z "$IP_ADDRS" ]; then 
    IP_ADDRS="127.0.0.1"
fi
HOSTNAME=`hostname -s`

# 收集如下指标
## CPU
metric_load=0
metric_tasks=0
metric_cpu_use=0
metric_cpu_wa=0

# 内存
metric_mem_use=0
metric_mem_total=0
metric_mem_use_prec=0

# 交换分区
metric_swap_use=0
metric_swap_total=0
metric_swap_use_prec=0

# 网络流量
declare -a metric_eths 
declare -a metric_eths_rx_bytes
declare -a metric_eths_tx_bytes

# 磁盘使用
declare -a metric_disks
declare -a metric_disks_use

# 磁盘IO
metric_io_use=0

top_info(){
    info=$(top -bn 1)
    if [ $DEBUG -eq 1 ]; then
        echo  "$info" | head -n5
        echo
    fi
    metric_load=`echo  "$info" | sed -n "1p" | grep -Eo "load average:.*"| awk '{print $3}'| sed 's/,//'`
    metric_tasks=`echo  "$info" | sed -n "2p" | grep -Eo ".* total"| grep -Eo "[0-9]+"`
    metric_cpu_use=`echo "$info" | sed -n "3p" | grep -Eo ": .*%us" | grep -Eo "[0-9]+\.?[0-9]*"`
    metric_cpu_wa=`echo "$info" | sed -n "3p" | grep -Eo "id,.*%wa" | grep -Eo "[0-9]+\.?[0-9]*"`
    metric_mem_use=`echo  "$info" | sed -n "4p" | grep -Eo ",.* used" | grep -Eo "[0-9]+"`
    metric_mem_total=`echo  "$info" | sed -n "4p" | grep -Eo ":.* total" | grep -Eo "[0-9]+"`
    metric_swap_use=`echo  "$info" | sed -n "5p" | grep -Eo ",.* used" | grep -Eo "[0-9]+"`
    metric_swap_total=`echo  "$info" | sed -n "5p" | grep -Eo ":.* total" | grep -Eo "[0-9]+"`
    
    let "metric_mem_use_prec=metric_mem_use*100/metric_mem_total"
    let "metric_swap_use_prec=metric_swap_use*100/metric_swap_total"

    if [ $DEBUG -eq 1 ]; then
        echo "load=$metric_load, tasks=$metric_tasks, cpu_use=$metric_cpu_use, cpu_wa=$metric_cpu_wa"
        echo "mem_use=$metric_mem_use, mem_total=$metric_mem_total, mem_use_prec=$metric_mem_use_prec"
        echo "swap_use=$metric_swap_use, swap_total=$metric_swap_total, swap_use_prec=$metric_swap_use_prec"
    fi
    echo
}

get_rx_bytes(){
    ifconfig $1 2>/dev/null | grep "RX bytes" \
        | grep -Eo "RX bytes:[0-9]+" | grep -Eo "[0-9]+"
}

get_tx_bytes(){
    ifconfig $1 2>/dev/null| grep "TX bytes"  \
        | grep -Eo "TX bytes:[0-9]+" | grep -Eo "[0-9]+"
}

eths_info(){
    metric_eths=`ifconfig | grep "Link encap" | cut -d" " -f1 | grep eth`
    i=0
    for eth in $metric_eths;do
        rx1=$(get_rx_bytes $eth)
        tx1=$(get_tx_bytes $eth)
        sleep 1;
        rx2=$(get_rx_bytes $eth)
        tx2=$(get_tx_bytes $eth)

        let "rx=rx2-rx1"
        let "tx=tx2-tx1"
        if [ $DEBUG -eq 1 ]; then
            echo $eth, rx1=$rx1, rx2=$rx2, rx=$rx
            echo $eth, tx1=$tx1, tx2=$tx2, tx=$tx
        fi

        metric_eths_rx_bytes[$i]=$rx
        metric_eths_tx_bytes[$i]=$tx

        i=`expr $i + 1`
    done
    echo
}

disk_info(){
    metric_disks=`df -h | grep -Eo "^/dev/.+da[0-9]" | sed -r "s/dev//g" | sed -r "s/\///g"`
    i=0
    for disk in $metric_disks;do
        disk_use=`df -h | grep -E "^/dev/${disk}" | grep -Eo "[0-9]+%" | grep -Eo "[0-9]+"`
        metric_disks_use[$i]=$disk_use
        if [ $DEBUG -eq 1 ]; then
            echo $disk, disk_use=$disk_use
        fi
        i=`expr $i + 1`
    done
    echo
}

io_info(){
    if [ -n "`type -p iostat`" ];then
        io_use=`iostat -c | sed -n "4p" | awk '{print $4}'`
        if [ $DEBUG -eq 1 ]; then
            echo io_use=$io_use
        fi
    fi
}

send_metric(){
    time=`date +%s`
    echo "$API_KEY/$HOSTNAME/$IP_ADDRS/$1 $2 $time"
    echo "$API_KEY/$HOSTNAME/$IP_ADDRS/$1 $2 $time" | nc $COLLECTOR_IP $COLLECTOR_PORT -w1>/dev/null
}

send_all(){
    send_metric "load" $metric_load
    send_metric "tasks" $metric_tasks
    send_metric "cpu_use" $metric_cpu_use
    send_metric "cpu_wa" $metric_cpu_wa

    send_metric "mem_use" $metric_mem_use
    send_metric "mem_total" $metric_mem_total
    send_metric "mem_use_prec" $metric_mem_use_prec

    send_metric "swap_use" $metric_swap_use
    send_metric "swap_total" $metric_swap_total
    send_metric "swap_use" $metric_swap_use_prec

    i=0
    for eth in $metric_eths;do
        send_metric "${eth}_rx_bytes" ${metric_eths_rx_bytes[$i]}
        send_metric "${eth}_tx_bytes" ${metric_eths_tx_bytes[$i]}
        i=`expr $i + 1`
    done

    i=0
    for disk in $metric_disks;do
        send_metric "${disk}_disk_use" ${metric_disks_use[$i]}
    done
    
    if [ -n "$metric_io_use" ];then
        send_metric "io_use" $metric_io_use 
    fi
}

collect() {
    echo time=`date +"%Y-%m-%d %H:%M:%S"` begin collect.
    echo "collector_ip=$COLLECTOR_IP, collector_port=$COLLECTOR_PORT, api_key=$API_KEY"
    echo "ip_addr=${IP_ADDRS}, hostname=$HOSTNAME"
    echo

    top_info
    eths_info
    disk_info
    io_info
    
    send_all
}

run(){
    trap "" HUP
    while :
    do
        sleep 60;
        collect >/dev/null 2>&1
    done;
}

collect
echo "DNSPod monitor agent run ok."
run &
