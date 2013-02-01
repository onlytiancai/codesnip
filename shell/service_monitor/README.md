### 服务监控脚本 

比如有如下的wsgi服务

    def app(environ, start_response):
          data = "Hello, World!\n"
          start_response("200 OK", [
              ("Content-Type", "text/plain"),
              ("Content-Length", str(len(data)))
          ])
          return iter([data])

如何让它长时间的运行，并且被kill掉或crash掉后能自动重启呢？

### 使用supervisor

最简单的办法就是使用supervisor和gunicorn

    pip install supervisor
    pip install gunicorn

然后在/etc/supervisord.conf下加入如下配置

    [program:service-monitor]
    command=gunicorn SimpleService:app -b localhost:7000
    directory=/home/huhao/src/github/codesnip/shell/service_monitor

然后用**supervisorctl start service-monitor**启动这个服务就行了。
这时候即使这个wsgi服务crash或者被认为kill掉了，supervisor都会自动给你起来。
supervisor还有更多的功能，详见官方文档，目前我们可以只用他这个功能。



创建日志文件并赋予权限

    sudo touch /var/log/SimpleService.monitor.log
    sudo chown `whoami` /var/log/SimpleService.monitor.log

### 使用shell

如果你是shell控，不喜欢用第三方软件，可以参考下面。

约定每个服务里要有 service.start.sh service.stop.sh service.status.sh 这几个脚本，分别用来启动，停止您的服务，以及查看您服务的状态。

service.start.sh 根据您的需要来编写，比如gunicron启动的服务就如下

    CPUCOUNT=`cat /proc/cpuinfo | grep processor | wc -l`
    gunicorn -w ${CPUCOUNT} SimpleService:app -b localhost:7000 -D -p SimpleService.pid

service.stop.sh 也是根据您的业务需要来编写，比如

    kill `cat SimpleService.pid`
    sleep 5

要查看一个服务是否运行正常，一般首先要保证相关的进程存在，以及监听的端口正常，然后这些逻辑都写在 service.status.sh 里，监控脚本会定时运行这个脚本，如果exit code是0的话就表示服务运行正常，否则的话就会调用 service.stop.sh 和 service.start.sh 去重启服务。

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

在crontab里设置每5分钟执行一次监控脚本

    */5 * * * * cd ~/service_monitor/ && bash ./service.monitor.sh >> /var/log/SimpleService.monitor.log

service.monitor.sh 是死的，不需要改动，如下。

    bash ./service.status.sh
    if [ $? -ne 0 ]; then
        echo  `date '+%Y-%m-%d %H:%M'` "service is not available";
        bash ./service.stop.sh
        bash ./service.start.sh
    else
        echo  `date '+%Y-%m-%d %H:%M'` "service is available";
    fi
