database:

    CREATE TABLE `feeds` (
      `id` int(11) NOT NULL AUTO_INCREMENT,
      `feed_id` bigint(20) NOT NULL DEFAULT '0',
      `author` varchar(32) NOT NULL DEFAULT '',
      `title` varchar(512) NOT NULL DEFAULT '',
      `summary` varchar(1024) NOT NULL DEFAULT '',
      `published` datetime NOT NULL DEFAULT '1970-01-01 00:00:00',
      PRIMARY KEY (`id`) /*T![clustered_index] CLUSTERED */,
      KEY `ix_feeds_published` (`published`),
      UNIQUE KEY `uq_feeds_feed_id` (`feed_id`)
    )

crontab

    0 * * * * cd /home/ubuntu/src/codesnip/python/hn_feeds && /usr/bin/python3 cron.py >/dev/null 2>&1 &

sqlite

    sudo apt install sqlite3
    sqlite3 hn.db

        .mode column
        .mode table

        CREATE TABLE `feeds` (
          `id` INTEGER PRIMARY KEY,
          `feed_id` INTEGER NOT NULL DEFAULT '0',
          `author` text NOT NULL DEFAULT '',
          `title` text NOT NULL DEFAULT '',
          `summary` text NOT NULL DEFAULT '',
          `published` text
        )

        .tables
        pragma table_info(feeds);

        CREATE INDEX idx_feeds_published ON feeds (published);
        CREATE UNIQUE INDEX idx_feeds_feed_id ON feeds (feed_id);
        PRAGMA index_list('feeds');
        .q

        alter table feeds add column main_content text;

run

    gunicorn -b 127.0.0.1:5001 -w 4 'app:app'

nginx

    server {
        listen 80;
        server_name  hn.ihuhao.com;

        access_log  /var/log/nginx/hn.access.log main;
        error_log /var/log/nginx/hn.error.log;
        location / {
            proxy_pass   http://127.0.0.1:5001;
            proxy_set_header    Host             $host;
            proxy_set_header    X-Real-IP        $remote_addr;
            proxy_set_header    X-Forwarded-For  $proxy_add_x_forwarded_for;
        }

    }

proxy

    curl -v http://6.6.6.6:3001/comply-or-withdraw -H 'Host: the.webm.ink' -H 'x-scheme: https'

    server {

        listen 3001;
        server_name _;

        proxy_connect_timeout 20s;
        proxy_read_timeout 20s;

        access_log /var/log/nginx/hn_proxy.access.log;
        error_log /var/log/nginx/hn_proxy.error.log;
        allow 140.143.188.251;
        allow 101.42.30.36;
        deny all;

        location / {
            resolver 127.0.0.53;
            proxy_pass $http_x_scheme://$http_host/;
            proxy_ssl_server_name on;
            proxy_intercept_errors on;
            error_page 301 302 307 = @handle_redirects;

            proxy_set_header User-Agent $http_user_agent;
        }

        location @handle_redirects {
            resolver 127.0.0.53;
            set $saved_redirect_location '$upstream_http_location';
            proxy_pass $saved_redirect_location;
            proxy_set_header User-Agent $http_user_agent;
        }
    }
