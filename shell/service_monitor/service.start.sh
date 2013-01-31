CURDIR=$(cd "$(dirname $0)"; pwd)
CPUCOUNT=`cat /proc/cpuinfo | grep processor | wc -l`
gunicorn -w ${CPUCOUNT} SimpleService:app -b localhost:7000 -D -p SimpleService.pid
