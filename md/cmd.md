启动带 cdp 的chrome

    /Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --remote-debugging-port=9222 --user-data-dir=/Users/huhao/chrome-profile-for-ai

mac 查看端口

    netstat -anp tcp | grep LISTEN

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
