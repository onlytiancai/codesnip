DEBUG=wawa-table npm start

supervisor bin/www

forever start bin/www
forever stop bin/www

使用forever运行nodejs应用
http://tcrct.iteye.com/blog/2043644

