CURDIR=$(cd "$(dirname $0)"; pwd)
CPUCOUNT=`cat /proc/cpuinfo | grep processor | wc -l`

function process_count(){
    return `ps -ef | grep "$1" | grep -v grep | wc -l`;
}

function port_is_ok(){
    return `netstat -npl 2>/dev/null | grep " $1 " | wc -l`;
}

# 测试进程是否都存在
process_count "SimpleService:app"
if [ ! $? -eq $((${CPUCOUNT}+1)) ] ; then
    echo "worker is down:$?";
    exit 1;
fi

# 测试端口是否正常监听
port_is_ok "127.0.0.1:7000"
if [ $? -eq 0 ]; then
    echo "port is down";
    exit 0;
fi

exit 0;
