API_KEY=${API_KEY:="test_api_key"}
COLLECTOR_IP=${COLLECTOR_IP:="127.0.0.1"}
COLLECTOR_PORT="2003"

IP_ADDRS=`ifconfig | grep 'inet addr' | grep -v '255.0.0.0' | cut -f2 -d':' | awk '{print $1}'`
IP_ADDRS=`echo $IP_ADDRS | sed 's/\n//g'`
HOSTNAME=`hostname -s`


function collect(){
    time=`date +%s`

    info=$(top -bn 1)
    echo  "$info" | head -n5
    load=`echo  "$info" | sed -n "1p" | grep -Eo "load average:.*"| awk '{print $3}'| sed 's/,//'`
    tasks=`echo  "$info" | sed -n "2p" | grep -Eo ".* total"| grep -Eo "[0-9]+"`
    cpu_use=`echo "$info" | sed -n "3p" | grep -Eo ": .*%us" | grep -Eo "[0-9]+\.?[0-9]*"`
    mem_use=`echo  "$info" | sed -n "4p" | grep -Eo ",.* used" | grep -Eo "[0-9]+"`
    swap_use=`echo  "$info" | sed -n "5p" | grep -Eo ",.* used" | grep -Eo "[0-9]+"`

    echo "collector_ip=$COLLECTOR_IP, collector_port=$COLLECTOR_PORT, api_key=$API_KEY"
    echo "load=$load, tasks=$tasks, cpu_use=$cpu_use, mem_use=$mem_use, swap_use=$swap_use, "
    echo "$API_KEY/$HOSTNAME/$IP_ADDRS/load $load $time"  | nc $COLLECTOR_IP $COLLECTOR_PORT 
    echo "$API_KEY/$HOSTNAME/$IP_ADDRS/tasks $tasks $time" | nc $COLLECTOR_IP $COLLECTOR_PORT 
    echo "$API_KEY/$HOSTNAME/$IP_ADDRS/cpu_use $cpu_use $time" | nc $COLLECTOR_IP $COLLECTOR_PORT 
    echo "$API_KEY/$HOSTNAME/$IP_ADDRS/mem_use $mem_use $time" | nc $COLLECTOR_IP $COLLECTOR_PORT 
    echo "$API_KEY/$HOSTNAME/$IP_ADDRS/swap_use $swap_use $time" | nc $COLLECTOR_IP $COLLECTOR_PORT 

}

function run(){
    while :
    do
        echo time=`date +"%Y-%m-%d %H:%M:%S"`, pid=$$
        collect;
        sleep 60;
    done;
}

run
