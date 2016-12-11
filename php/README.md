nginx php-fpm安装配置
http://www.nginx.cn/231.html

yum -y install gcc automake autoconf libtool make

yum -y install gcc gcc-c++ glibc

yum -y install libmcrypt-devel mhash-devel libxslt-devel \
libjpeg libjpeg-devel libpng libpng-devel freetype freetype-devel libxml2 libxml2-devel \
zlib zlib-devel glibc glibc-devel glib2 glib2-devel bzip2 bzip2-devel \
ncurses ncurses-devel curl curl-devel e2fsprogs e2fsprogs-devel \
krb5 krb5-devel libidn libidn-devel openssl openssl-devel



wget http://php.net/get/php-5.5.27.tar.bz2/from/a/mirror
./configure --prefix=/usr/local/php  --enable-fpm --with-mcrypt \
--enable-mbstring --disable-pdo --with-curl --disable-debug  --disable-rpath \
--enable-inline-optimization --with-bz2  --with-zlib --enable-sockets \
--enable-sysvsem --enable-sysvshm --enable-pcntl --enable-mbregex \
--with-mhash --enable-zip --with-pcre-regex --with-mysql --with-mysqli \
--with-gd --with-jpeg-dir

make
make test
make install

cd /usr/local/php
cp etc/php-fpm.conf.default etc/php-fpm.conf
vi etc/php-fpm.conf


ini_set('display_errors', 'On');
error_reporting(E_ALL &~ E_NOTICE)

CentOS6.5下添加epel源
http://www.centoscn.com/CentOS/config/2014/0920/3793.html
CentOS 6.2 yum安装配置lnmp服务器(Nginx+PHP+MySQL)
http://www.osyunwei.com/archives/2353.html

composer config -g repo.packagist composer https://packagist.phpcomposer.com

sudo vi /etc/php.ini
date.timezone = "UTC"

https://packagist.org/packages/hassankhan/config

https://packagist.org/packages/justinrainbow/json-schema

https://packagist.org/packages/respect/validation
https://packagist.org/packages/beberlei/assert
https://packagist.org/packages/nikic/fast-route
https://packagist.org/packages/acquia/http-hmac-php


    server {
        listen       80;
        server_name  php.ihuhao.com;
        root /home/wawa/src/phpci;

        location / {
            try_files $uri $uri/ /index.php$is_args$args;
        }

        location ~ \.php$ {
            try_files $uri = 404;
            fastcgi_pass   127.0.0.1:9000;
            fastcgi_index  index.php;
            include        fastcgi_params;
            fastcgi_param SCRIPT_FILENAME $document_root$fastcgi_script_name;
            fastcgi_param SERVER_NAME $http_host;
            fastcgi_ignore_client_abort on; 
            fastcgi_connect_timeout 600s;
            fastcgi_send_timeout 600s;
            fastcgi_read_timeout 600s;
        }  

    }


http://stackoverflow.com/questions/1921421/get-the-first-element-of-an-array
