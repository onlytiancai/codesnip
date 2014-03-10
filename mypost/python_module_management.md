# python 模块管理

## 面临问题

1. 单台服务器上运行多个python服务，每个服务依赖不同的模块，如何对项目运行环境进行隔离。
1. 公司内部会开发一些通用的python模块，但没有开源，如何像使用第三方模块那样用pip安装这些模块。

## 使用virtualenv隔离运行环境

每个python服务都使用独立的virtual_env环境。

    virtualenv --no-site-packages ~/.test_env   # 创建virtual_env环境

每个virtual_env环境里的一个特定的模块只安装一个版本，尽管site-packages目录下可以一个模块多版本共存，但这样管理起来会复杂。

## 使用pypiserver建立私有pypi

考察了几个可以建立私有pypi的包，[pypiserver](http://pypi.python.org/pypi/pypiserver/0.6.1)是最轻量给力的一个，也很灵活，能很好的和各种webserver搭配使用。

有一个问题就是这个简单的pypi服务器不支持密码认证，要想增加访问限制的话，最简单的办法就是让pypiserver监听一个本地端口，完了由nginx做反向代理对外发布出去，并在nginx上进行ip白名单的限制。

在生产环境找一台机器，把pypiserver搭建起来。

    setsid pypi-server -p 8085 -i 192.168.1.1 ./packages 1>/dev/null 2>&1 & #启动pypiserver

完了把公司内部的私有模块打包后方到packages目录下, 然后这个机房内的所有机器就可以使用pip安装私有的模块了。

## 使用pip安装服务的依赖

每个python服务的根目录下要放置一个requirements.txt文件, 如果服务是根据[pyempty](https://github.com/onlytiancai/pyempty)创建的，已经内置了这个文件,如下是一个示例文件。

    gevent==1.0dev
    web.py>=0.36

关于 requirements.txt的书写规范，可[参考这里](http://www.pip-installer.org/en/latest/requirements.html)。

virtual_env环境里内置了pip，直接使用这个pip安装该项目依赖的模块。

    . ~/.test_env/bin/activate                                          # 加载virtual_env环境
    pip install -i http://192.168.1.1:8085/simple/ -r requirements.txt  #根据依赖文件自动安装指定版本的模块

## 依赖特定版本的模块-不建议使用

为了防止模块升级后影响大量的项目，项目对模块的依赖要明确版本，具体如下：

    用easy_install安装的包都会写在easy-install.pth里，这时候一个包只能使用一个版本，
    easy_install -m module_name后该包信息就在.pth里去掉了，然后这个包就可以多版本并存了，
    然后pkg_resources.require('wawa==1.0')后import wawa就可以在代码中强制使用指定版本的包了
