DEBUG=wawa-table npm start

supervisor bin/www

forever start bin/www
forever stop bin/www

使用forever运行nodejs应用
http://tcrct.iteye.com/blog/2043644

## todo
- 日志记录request_id, file, line_no, client_ip, 
- API测试
