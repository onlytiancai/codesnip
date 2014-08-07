独立日志

    logfile = '/data/log/login_mon/%s.log' % datetime.now().strftime('%Y-%m-%d')
    logging.basicConfig(filename=logfile, level=logging.INFO, format='%(asctime)s - %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())

    def init_logger(log_filename, level='info', console=False):
        import logging.handlers
        logger = logging.getLogger()
        logger.propagate = False
        level = logging._levelNames.get(level.upper(), logging.INFO)
        logger.setLevel(level)
        handler = logging.handlers.RotatingFileHandler(log_filename, maxBytes=100 * 1000 * 1000, backupCount=10)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        if console:
            consoleHandler = logging.StreamHandler()
            consoleHandler.setFormatter(formatter)
            logger.addHandler(consoleHandler)
        logger.addHandler(handler)



异常日志记录器

    import functools
    import logging
    def log_exception(op_name, rethrow=True):
        '自动记录异常信息的修饰器，op_name:操作名称，rethrow：捕获异常后是否重新抛出'
        def inner(fun):
            @functools.wraps(fun)
            def inner2(*args, **kargs):
                try:
                    return fun(*args, **kargs)
                except:
                    logging.exception("%s error:args=%s, kargs=%s", op_name, args, kargs)
                    if rethrow:
                        raise
            return inner2
        return inner
                                                                                      
    if __name__ == "__main__":
        @log_exception("foo")
        def foo(a, b=2):
            raise Exception("heihei")
                                                                                      
        foo(1, b=100)

web.py骨架代码
    
    // ====== webmain.py
    # -*- coding: utf-8 -*-
    import os
    import web

    tpl_dir = os.path.join(os.path.dirname(__file__), 'templates')
    render = web.template.render(tpl_dir, base='layout')


    class index(object):
        def GET(self):
            return render.index()

    urls = ["/", index,
            ]

    app = web.application(urls, globals())
    wsgiapp = app.wsgifunc()

    if __name__ == '__main__':
        app.run()
    
    # 调试启动：python webmain.py 0.0.0.0:8000
    # 正式启动：gunicorn webmain:wsgiapp -b 0.0.0.0:8000 -w 4 -D

    // ====== templates/layout.html
    $def with (content)
    <!DOCTYPE html>
    <html lang="zh-cn">
        <head>
            <meta charset="utf-8">
            <title>$content.get('title', 'My site')</title>
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <link href="http://libs.baidu.com/bootstrap/3.2.0/css/bootstrap.css" rel="stylesheet">
            <script src="http://libs.baidu.com/jquery/1.8.2/jquery.js"></script>
            <script src="http://libs.baidu.com/bootstrap/3.2.0/js/bootstrap.js"></script>
        </head>
        <body>
            <div class="navbar navbar-fixed-top navbar-default" role="navigation">
                <div class="container">
                    <div class="navbar-header">
                        <button class="navbar-toggle collapsed" type="button" data-toggle="collapse" data-target=".navbar-collapse">
                            <span class="sr-only">切换导航</span>
                            <span class="icon-bar"></span>
                            <span class="icon-bar"></span>
                            <span class="icon-bar"></span>
                        </button>
                        <a class="navbar-brand" href="/">我的网站</a>
                    </div>
                    <div class="navbar-collapse collapse">
                        <ul class="nav navbar-nav">
                            <li class="active"><a href="/">首页</a></li>
                        </ul>
                        <ul class="nav navbar-nav navbar-right">
                            <li><a href="/about">关于</a></li>
                        </ul>
                    </div>
                </div>
            </div>
            <div class="container" style="margin-top:60px;min-height:500px;">
            $:content
            </div>
            <footer>
            <div class="container">
                <p>&copy; company 2014.</p>
            </div>
            </footer>
        </body>
    </html>

    // ====== templates/index.html
    $var title: This is title.

    <h3>Hello, world</h3>



去掉requests的默认日志

    logging.getLogger("requests.packages.urllib3").setLevel(logging.CRITICAL)  

使用MYSQL

    class Database:
        '''
        >>> db = Database(dict(host='localhost', user='root', passwd='password', db='information_schema'))
        >>> sql = 'select TABLE_NAME from TABLES'
        >>> result = db.fetchall(sql, ())
        >>> len(result) > 0
        True
        '''
        conns = {}

        class Cursor:
            def __init__(self, conn):
                self.cursor = conn.cursor()

            def __enter__(self):
                return self.cursor
                        
            def __exit__(self, e_t, e_v, t_b):
                self.cursor.close()

        def __init__(self, conn_args):
            self.conn_args = conn_args

        def _get_conn(self):
            '''
            如果连接未建立，建立连接并缓存
            如果连接被异常关闭，则重新连接
            '''
            sort_args = tuple("%s=%s" % (k, self.conn_args[k])
                              for k in sorted(self.conn_args))
            if sort_args not in Database.conns:
                Database.conns[sort_args] = MySQLdb.connect(**self.conn_args)
            else:
                try:
                    Database.conns[sort_args].ping()
                except:
                    Database.conns[sort_args] = MySQLdb.connect(**self.conn_args)

            return Database.Cursor(Database.conns[sort_args])

        def fetchall(self, sql, args):
            with self._get_conn() as conn:
                conn.execute(sql, tuple(args))
                return list(conn.fetchall())

        def excute(self, sql, args):
            with self._get_conn() as conn:
                conn.execute(sql, tuple(args))
                conn.commit()
