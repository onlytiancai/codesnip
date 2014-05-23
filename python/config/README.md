一个服务会运行在各种环境，如开发环境，测试环境，生产环境，
每个环境有不同的配置，为了合理的管理这些配置项，做如下约定

- 默认配置项保存在config.py里，受git管理
- 每个环境有一个独立的config_xxx.py文件，不受git管理, xxx表示环境名
- config_xxx.py里的配置会覆盖掉config.py的配置

不同环境启动程序时修改下`APP_ENV`的环境变量，从而加载不同的配置，如

    APP_ENV=dev python app.py
    APP_ENV=testing python app.py
    APP_ENV=production python app.py
