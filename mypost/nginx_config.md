目的：

1. 为了让一个主机IP支持多个https域名，需要启用SNI, 需要重新编译openssl和指定--with-openssl参数
1. 为了让nginx使用lua脚本，需要打开--with-luajit开关
1. 为了能查看nginx性能，需要打开http_stub_status模块

最终安装脚本如下

    # wget openssl, tar xf
    cd openssl-1.0.1e
    ./config enable-tlsext
    make -j8
    make install

    # wget openresty, tar xf
    ngx_openresty-1.2.6.6
    ./configure --prefix=/usr/local/openresty \
                --with-luajit \
                -–with-http_stub_status_module \
                --with-openssl=../openssl-1.0.1e/ \
                -j2
    make -j9
    make install

    # will output "TLS SNI support enabled"
    /usr/local/openresty/nginx/sbin/nginx -V

nginx示例配置

    worker_processes  9;
    worker_rlimit_nofile 30000;

    error_log  /data/log/nginx/error.log;

    events {
        worker_connections  51200;
        use epoll;
    }


    http {
        include       mime.types;
        default_type  application/octet-stream;

        log_format  main  '$remote_addr - $remote_user [$time_local] "$request" '
                          '$status $body_bytes_sent "$http_referer" '
                          '"$http_user_agent" "$http_x_forwarded_for"';

        access_log  /data/log/nginx/access.log  main;

        sendfile        on;

        keepalive_timeout  65;

        gzip on;
        gzip_http_version 1.1;
        gzip_comp_level 2;
        gzip_types    text/plain text/css application/x-javascript text/xml text/javascript;

        server {
            listen       80;
            listen  443 ssl;
            server_name  huhao.me;
            ssl_certificate /usr/local/openresty/nginx/ssl/huhao.me.crt;
            ssl_certificate_key /usr/local/openresty/nginx/ssl/huhao.me.key;

            location / {
                proxy_set_header X-Real-IP $remote_addr;
                proxy_set_header REMOTE-HOST $remote_addr;
                proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
                proxy_pass   http://localhost:7001;
            }

            location /static/ {
                alias /root/src/robin/main/static/;
                expires 1h;
            }

            location /nginx_status {
                stub_status on;
            }
        }

        server {
            listen       80;
            listen  443 ssl;
            server_name  blog.huhao.me;
            ssl_certificate /usr/local/openresty/nginx/ssl/blog.huhao.me.crt;
            ssl_certificate_key /usr/local/openresty/nginx/ssl/huhao.me.key;

            location / {
                proxy_set_header X-Real-IP $remote_addr;
                proxy_set_header REMOTE-HOST $remote_addr;
                proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
                proxy_pass   http://localhost:7002;
            }
        }

    }
