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
    load=`echo  "$info" | sed -n '1p' | sed -r 's/.*load average:[[:space:]]([0-9]+).*/\1/'`
    tasks=`echo  "$info" | sed -n '2p' | awk '{print $2}'`
    cpu_use=`echo  "$info" | sed -n '3p' | awk '{print $2}' | sed -r 's/([0-9]+).*/\1/'`
    mem_use=`echo  "$info" | sed -n '4p' | awk '{print $4}' | sed -r 's/([0-9]+).*/\1/'`
    swap_use=`echo  "$info" | sed -n '5p' | awk '{print $4}' | sed -r 's/([0-9]+).*/\1/'`

    echo "collector_ip=$COLLECTOR_IP, collector_port=$COLLECTOR_PORT, api_key=$API_KEY"
    echo "load=$load, tasks=$tasks, cpu_use=$cpu_use, mem_use=$mem_use, swap_use=$swap_use, "
    echo "$API_KEY/$HOSTNAME/$IP_ADDRS/load $load $time"  | nc $COLLECTOR_IP $COLLECTOR_PORT 
    echo "$API_KEY/$HOSTNAME/$IP_ADDRS/tasks $tasks $time" | nc $COLLECTOR_IP $COLLECTOR_PORT 
    echo "$API_KEY/$HOSTNAME/$IP_ADDRS/cpu_use $cpu_use $time" | nc $COLLECTOR_IP $COLLECTOR_PORT 
    echo "$API_KEY/$HOSTNAME/$IP_ADDRS/mem_use $mem_use $time" | nc $COLLECTOR_IP $COLLECTOR_PORT 
    echo "$API_KEY/$HOSTNAME/$IP_ADDRS/swap_use $swap_use $time" | nc $COLLECTOR_IP $COLLECTOR_PORT 

}

while :
do
    echo `date`
    collect;
    sleep 60;
done;
