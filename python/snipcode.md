独立日志

    import logging                                                                 

    log_filename = '/var/log/monitor-feedback.log'                                 

    logger = logging.getLogger('monitor.feedback')                                    
    logger.propagate = False                                                          
    logger.setLevel(logging.DEBUG)                                                    
        handler = logging.handlers.RotatingFileHandler(log_filename, maxBytes=100 * 1000 * 1000, backupCount=10)
        formatter = logging.Formatter('%(asctime)s - %(message)s')                        
        handler.setFormatter(formatter)                                                   
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

    # -*- coding: utf-8 -*-
    import web

    class index(object):
        def GET(self):
            return "Hello word."

    urls = [ 
            "/", index,
           ]

    app = web.application(urls, globals())
    wsgiapp = app.wsgifunc()

    if __name__ == '__main__':
        app.run()
    
    # 调试启动：python testweb.py 0.0.0.0:8000
    # 正式启动：gunicorn testweb:wsgiapp -b 0.0.0.0:8000 -w 4 -D

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
