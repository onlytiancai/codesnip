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
