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
