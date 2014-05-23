# -*- coding: utf-8 -*-
'''
>>> import sys
>>> config = sys.modules[__name__]
>>> config.app_name
'default_app'
>>> import os; os.environ['APP_ENV'] = 'dev'
>>> reload_config()
>>> config.app_name
'dev_app'
'''

# 以下为服务运行需要的各种默认配置项
app_name = 'default_app'


# 根据不同的环境动态的覆盖默认配置项
def reload_config():
    import os
    app_env = os.environ.get('APP_ENV')
    if app_env:
        app_config = __import__('config_%s' % app_env)
        items = [(k, v) for k, v in app_config.__dict__.items()
                 if not k.startswith('__')]
        globals().update(dict(items))

reload_config()

if __name__ == '__main__':
    import doctest
    doctest.testmod()
