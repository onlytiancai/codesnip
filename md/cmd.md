启动带 cdp 的chrome

    /Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --remote-debugging-port=9222 --user-data-dir=/Users/huhao/chrome-profile-for-ai

查看端口

    # mac
    netstat -anp tcp | grep LISTEN

    # 查看 udb 绑定端口
    sudo ss -unlp
    # 查看 tcp 监听端口
    sudo ss -tnlp

启动 ttyd 共享终端

    ttyd -i "$(tailscale ip -4)" -p 7681  -W tmux new-session -A -s claude

抓网页转换成 markdown

    node ~/src/codesnip/node/html-to-md/src/index.js -u https://hacker-podcast.agi.li/episode/2026-10-30|python -m rich.markdown -

查看单词

    python /Users/huhao/src/codesnip/python/ai/024-cet4/scripts/cet4.py

vmstat 1 看磁盘压力

    wa 是 CPU 的 I/O wait 百分比。
    - wa 接近 0：CPU 基本没有明显等待 I/O。
    - wa 持续较高：系统可能存在 I/O 等待。
    - wa 高，同时磁盘 await 和 aqu-sz 也高：比较像磁盘瓶颈。
    注意：wa 高不一定完全是磁盘造成的，还要排查 NFS、存储网络等 I/O 问题。

iostat 看磁盘压力

    $ iostat -xz 1| awk 'NF>10&&$1!~/loop/ {printf "%-10s %10s %10s %10s %10s %10s %10s %10s %10s\n", $1,$2,$3,$6,$8,$9,$12,$22,$23}'
    Device            r/s      rkB/s    r_await        w/s      wkB/s    w_await     aqu-sz      %util
    md2              0.00       0.00       0.08       0.00       0.04      18.23       0.00       0.00
    md3           3504.25   88473.13       0.11    7280.56  112517.60       0.06       0.09      35.37
    nvme0n1        489.43   19383.66       0.12    3667.67   43200.71       0.07       0.31       9.47
    nvme1n1       1286.19   31270.63       0.11    4146.51   72018.13       0.07       0.44      15.49
    nvme2n1        491.20   19381.78       0.12    4151.50   72018.11       0.07       0.35      10.45
    nvme3n1       1283.11   31216.55       0.11    3662.88   43200.87       0.07       0.40      14.67

多进程并行

    $ seq 1 1000 | xargs -n 1 -P 10 -I {} python worker.py {}
    
    $ cat ranges.txt
    1 100
    101 200
    201 300
    ...
    901 1000
    $ cat ranges.txt | xargs -n 2 -P 10 ./worker.sh

按第三列倒序排，第三列是浮点数字

    sort -k3,3nr file.txt

claude 滚动条修复

    /tui default
    /tui fullscreen
    /terminal-setup

复制 screen 输出的文本

    screen -S session_name -X hardcopy -h /tmp/screen.log

screen 日志

    screen -S hao -X logfile /tmp/my_screen.log
    screen -S hao -X log on
    tail -f /tmp/my_screen.log
    screen -S hao -X log off
测试 ollama API

    curl -s http://localhost:11434/api/generate \
      -d '{
        "model": "s2021008840/hy-mt2:1.8b",
        "prompt": "Translate Chinese to English: 我喜欢人工智能。",
        "stream": false
      }' | jq -r '.response'

去掉代理使用 curl

    curl -s --noproxy '*' http://localhost:11434/api/tags

duckdb 把csv转parquet

    COPY (
        SELECT
            id::BIGINT AS id,
            parent_id::BIGINT AS parent_id,
            is_parent::TINYINT AS is_parent,
            file_id::INTEGER AS file_id
        FROM read_csv(
            '202611.csv',
            header = false,
            columns = {
                'id': 'BIGINT',
                'parent_id': 'BIGINT',
                'is_parent': 'TINYINT',
                'file_id': 'INTEGER'
            }
        )
    ) TO '202611.parquet'
    (FORMAT PARQUET, COMPRESSION ZSTD);

jsonl 处理

    jq 'select(.id==517115)|.data' out.jsonl

git 撤回最近一次提交里的一个文件

    # 撤回最后一次 commit，保留暂存区
    git reset --soft HEAD~1

    # 将不想提交的文件从暂存区移除
    git restore --staged 文件名

gocryptfs 加密文件夹

    sudo apt update
    sudo apt install gocryptfs

    gocryptfs -version
    ls -l /dev/fuse

    mkdir -p ~/secret.encrypted
    mkdir -p ~/secret.mount

    gocryptfs -init ~/secret.encrypted

    gocryptfs ~/secret.encrypted ~/secret.mount
    gocryptfs -i 10m ~/secret.encrypted ~/secret.mount

    echo "My secret information" > ~/secret.mount/password.txt
    mkdir ~/secret.mount/private
    echo "Sensitive data" > ~/secret.mount/private/data.txt

    fusermount3 -u ~/secret.mount

    vim ~/.bash_logout
    # Automatically unmount gocryptfs on logout
    if mountpoint -q "$HOME/secret.mount"; then
        fusermount3 -u "$HOME/secret.mount"
    fi

python uv

    curl -LsSf https://astral.sh/uv/install.sh | sh
    uv self update
    uv init
    uv python install 3.12
    uv python pin 3.12

    uv python list --only-installed

    uv add numpy pandas matplotlib
    uv add --dev pytest ruff
    uv tree
    uv remove pandas
    uv sync
    uv pip list
    uv pip freeze

    uv run python main.py
    uv run pytest
    uv run ruff check .

    uv venv
    uv venv --python 3.12
    source .venv/bin/activate
    deactivate

    uv add requests # 项目管理模式，会维护项目依赖声明；
    uv pip install requests # pip 兼容模式，主要用于直接管理虚拟环境中的包。

    uv tool list
    uv tool install ruff
    uv tool uninstall ruff
    uvx ruff check .

ssh 禁止密码登录

    sudo vim /etc/ssh/sshd_config

    PasswordAuthentication no
    KbdInteractiveAuthentication no
    PubkeyAuthentication yes
    PermitEmptyPasswords no

    sudo sshd -t
    sudo sshd -T | grep -E 'passwordauthentication|kbdinteractiveauthentication|pubkeyauthentication'
    sudo systemctl restart ssh

带进度多进程压缩

    pv largefile.csv | pigz -p $(nproc) > compressed.csv.gz

给 ssh 私钥加密码，减少密钥泄漏被利用风险

    ssh-keygen -p -f ~/.ssh/id_rsa
    ssh-keygen -p -f ~/.ssh/id_ed25519

ZeroTier 使用

    ip -d link show ztwfuexqwn
    ps aux | grep -Ei 'vpn|wireguard|zerotier|tailscale|openvpn|sing-box|clash|mihomo'
    ip route
    ip rule
    ethtool -i ztwfuexqwn

    sudo zerotier-cli status
    sudo zerotier-cli listnetworks
    sudo zerotier-cli peers
    sudo zerotier-cli info
    sudo zerotier-cli join <NETWORK_ID>
    sudo zerotier-cli leave <NETWORK_ID>


卸载 ZeroTier

    sudo systemctl stop zerotier-one
    sudo systemctl disable zerotier-one
    sudo apt purge zerotier-one -y
    sudo apt autoremove -y
    sudo rm -rf /var/lib/zerotier-one
    pgrep -a zerotier
    ip link show | grep -i zt
    systemctl status zerotier-one

find 

    # 排除某些目录名
    find ./ -type d -name "node_modules" -prune -o -name "*.md" -print
    # 排除某目录
    find . -path ./server -prune -o -path ./node_modules -prune -o -name '*.md'

ufw
    sudo ufw allow 22

certbot

    sudo apt install certbot
    sudo apt install python3-certbot-nginx -y
    sudo certbot plugins
    sudo certbot --nginx    
    sudo ls /etc/letsencrypt/live/
    sudo certbot certificates
    sudo certbot renew --dry-run
    sudo systemctl status certbot.timer
    sudo systemctl status certbot.service
    apt list --installed 2>/dev/null | grep certbot

nginx 设置密码

    sudo apt install apache2-utils
    sudo htpasswd -c /etc/nginx/.htpasswd admin 
    sudo htpasswd /etc/nginx/.htpasswd user2

    # nginx 配置
    auth_basic "Restricted Access";
    auth_basic_user_file /etc/nginx/.htpasswd;

tcpdump

    # 发送端抓包
    sudo tcpdump -i any -nn 'udp and dst host <接收端IP> and dst port 8125'
    # 接收端抓包
    sudo tcpdump -i any -nn -A udp port 8125
    sudo tcpdump -i any -nn 'udp and dst port 8125'
    sudo tcpdump -i any -nn 'host <发送端公网IP> and udp port 8125'

statsD 测试发送

    echo "test.counter:1|c" | nc -u -w1 192.168.1.100 8125

路由查看

    ip route get 124.237.177.164

iptables
    
    # 是否有丢包
    sudo iptables -vnL INPUT --line-numbers

docker

    sudo apt update
    sudo apt install -y docker.io docker-compose-v2

    sudo systemctl enable --now docker
    docker --version
    docker compose version    

    sudo docker compose up -d
    sudo docker compose ps
    sudo docker logs -f graphite
    docker exec -it graphite bash

apt-get 自动更新锁问题检查

    sudo tail -100 /var/log/apt/term.log
    sudo tail -100 /var/log/unattended-upgrades/unattended-upgrades.log

    ps -p 1038256 -o pid,ppid,stat,etime,time,%cpu,%mem,cmd

    sudo lsof /var/lib/dpkg/lock-frontend
    sudo lsof /var/lib/dpkg/lock

    sudo systemctl stop unattended-upgrades
    sudo apt-get -f install
    sudo dpkg --configure -a
    sudo apt-get update

    sudo systemctl status unattended-upgrades
    sudo systemctl disable --now unattended-upgrades

    ps -o pid,ppid,stat,wchan:32,cmd -p 948435,142761
    sudo cat /proc/948435/status | grep -E 'State|SigPnd|ShdPnd'
    sudo cat /proc/948435/wchan
    sudo pstree -ap 948435
    systemctl list-jobs
    sudo systemctl --failed

    sudo systemctl cat nginx.service
    sudo systemctl show nginx.service \
      -p Type \
      -p NotifyAccess \
      -p TimeoutStartUSec \
      -p ExecStart \
      -p MainPID \
      -p ActiveState \
      -p SubState
查看日志

    sudo journalctl -u unattended-upgrades --since "7 days ago"
