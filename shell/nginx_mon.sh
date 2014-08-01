api_key='xxx'
nginx_log_path="/usr/local/nginx/logs/dnsapi.access.log"
hostname='host1'
ip_addrs='127.0.0.1'

COLLECTOR_IP=${COLLECTOR_IP:="collector.monitor.dnspod.cn"}
COLLECTOR_PORT="2003"
send_metric(){
    time=`date +%s`
    echo "$api_key/$hostname/$ip_addrs/$1 $2 $time"
    echo "$api_key/$hostname/$ip_addrs/$1 $2 $time" | nc collector.monitor.dnspod.cn 2003 -w1>/dev/null
}


# 前一分钟的时间, 格式为linux日志的时间格式
time=`date -d "-1 minutes" +%d/%b/%Y:%H:%M`
echo ${time}

count_all=`tail -n 100000 ${nginx_log_path} | grep "${time}" | wc -l`
count_2xx=`tail -n 100000 ${nginx_log_path} | grep "${time}.*\"2[0-9][0-9]\"" | wc -l`
count_3xx=`tail -n 100000 ${nginx_log_path} | grep "${time}.*\"3[0-9][0-9]\"" | wc -l`
count_4xx=`tail -n 100000 ${nginx_log_path} | grep "${time}.*\"4[0-9][0-9]\"" | wc -l`
count_5xx=`tail -n 100000 ${nginx_log_path} | grep "${time}.*\"5[0-9][0-9]\"" | wc -l`

send_metric "count_all" $count_all
send_metric "count_2xx" $count_2xx
send_metric "count_3xx" $count_3xx
send_metric "count_4xx" $count_4xx
send_metric "count_5xx" $count_5xx
