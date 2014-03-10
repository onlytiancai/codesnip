## python项目通用组件和基础服务

很多公司都大量使用了python，其中有一些开发规范，code guidline， 通用组件，基础框架是可以共用的。

每个公司都自己搞一套, 太浪费人力，我想开一帖和大家讨论一下这些python基础设施的搭建。

原则是我们尽量不重新发明轮子，但开源组件这么多，也要有个挑选的过程和组合使用的过程，在这里讨论一下。

另一方面，有些开源组件虽然强大，但我们不能完全的驾驭它，或只使用其中很少的一部分，我们就可以考虑用python实现一个简单的轮子，可控性更强，最好不要超过300行代码。

### 规范开发流程

分为模块开发和项目开发

1. 模块: 实现多项目通用的功能，如日志，通信，性能计数等，并且要有良好的版本管理，直接安装到virtual_env环境里。
1. 项目: 实现具体的业务逻辑，依赖一些通用的模块，并以进程的形式存在。

关于通用模块的开发规范，我草拟了一份，如下：
    [如何编写高质量的python程序](https://github.com/onlytiancai/codesnip/blob/master/mypost/How_to_write_high-quality_python_program.md)
关于生产环境模块的管理规范，我也草拟了一下，如下
    [python 模块管理](https://github.com/onlytiancai/codesnip/blob/master/mypost/python_module_management.md)

关于项目的开发规范，因为涉及到部署，备份，日志存放，配置文件等很多运行时要考虑的东西，还没想出一套最佳的方法。

### 运维自动化

项目上线后还有很多事情要做，基础模块更新，项目代码更新，配置更新，代码备份，数据备份，性能监控，业务监控，日志分析，问题排查等等很多事情要做，都去手工做的话，非常费劲，所以要把这些事情向自动化的方向去考虑，最起码要工具化。

假设一个机房有一个中心化的SOCenter，每台机器上有一个SOAgent，SOCenter可以通过内部通信给SOAgent发送各种管理指令，SOAagnet也会定时上报这台机器上所有服务的状态，报警等信息。

大多数单项的功能我们都做好了，但还没合理的组合起来，这里主要画一下饼。

#### 服务启动，停止，重启

首先，每个服务的启动，停止，重启要规范化，一般linux里的服务都是以下方式来进行，所以我们自己的服务也要搞成这样的，进入到机器后，直接通过命令就可以完成这些操作。

    service myapp start
    service myapp stop
    service myapp restart

另外，这些操作还要提供api的支持，以支持SOCenter 集中化的管理，根本就不用登录这台机器。

    http://localhost/services/myapp/start
    http://localhost/services/myapp/stop
    http://localhost/services/myapp/restart

#### 服务运行情况检查
    
1. 每个服务要在元数据里声明自己要监听的端口，SOAgent会自动探测这些端口是否正常，异常时自动报警给SOCenter。
1. SOAgent还会定时ping每个服务，如果出现异常也报警给SOCenter，这就要求每个服务要实现接受ping的接口。

### 日志存储/日志查看

1. 每个服务的日志存储要规范化，存储到固定的指定的位置，SOAgent可以知道每个服务的日志在哪里。
1. SOCenter可以给SOAgent下发指令，在某个服务的日志里grep一个关键字并返回结果，用于远程问题排查。
1. 文本日志要限制大小和文件滚动规则，防止把硬盘撑满。

### 业务监控

基本的CPU，网络，内存，磁盘IO等机器资源的监控，都有开源的工具去做，但每个服务自身的各种业务指标，也要监控起来。

这就要让每个服务通过一个基础组件，定义自己的业务指标，发送给本机的SOAgent，并上报汇总给SOCenter。

这个我们已经做好了，接口非常简单，模仿windows的性能计数器接口，使用起来只要定义一个计数器，然后increment就行了。

    test_counter = counter.Counter('test')
    test_counter.increment() 

服务里定义和使用了计数器后，SOAgent会自动定时给SOCenter上报数据，SOCenter 可以查看各种业务指标报警和详细图表。

### 获取配置/配置更新

每个服务的配置一定要集中化，不可能每个服务都用自己的本地配置文件。

服务通过SOAgent 去SOCenter 获取自己的配置，前提是要传递机器名称，服务名称。

SOCenter还可以给SOAgent发送指令，对某个服务的配置进行热更新，这需要服务里实现一个接受配置变化的接口。

### 更新代码

项目新版本上线，需要处理很多台机器的模块升级，项目代码更新，代码备份，失败回滚，部署结果生成报表等。

SOCenter 可以看到每个服务都部署到了哪些机器上，并且可以看到每个服务的版本号，依赖的基础模块的版本号。

SOCenter 可以给SOAgent 下发部署指令，SOAgent会执行如下操作：

1. 备份服务代码到指定目录
1. 根据部署清单，升级基础模块到virtual_env环境
1. 根据部署清单，从SOCenter下载deploy build的tar.gz包
1. 停止服务
1. 解压build包，更新服务代码
1. 启动服务
1. 根据服务的元数据，执行一段脚本，验证服务是否启动正常，并生成部署报告返回给SOCenter

如果部署失败，SOCenter 可以给SOAgent 下发指令，把基础模块和服务代码回滚到上个版本。

### 备份

1. 服务通过元数据描述自己需要备份哪些文件，以及备份策略
1. SOAgent会根据这些规则定时备份服务的代码
1. SOCenter可以查看所有机器所有服务的备份清单

### 通用组件

因为内部的一些业务模块使用了gevent，而且好多开源的东西都有自己的event loop，和gevent搭配不起来，所以有一些组件不能直接用开源产品。

#### 通信组件

各个服务之间肯定是需要各种各样的内网通信的，[云风的这篇文章](http://blog.codingnow.com/2011/02/zeromq_message_patterns.html)进行了总结，但我觉得request/response模式是最有用的模式，其它的可以扩展出来。

我们基于这个思想开发了一个简单的通信组件：[xiwangshe](https://github.com/onlytiancai/xiwangshe)

备选：[celery](http://pypi.python.org/pypi/celery) [zeromq](http://www.zeromq.org/)

#### 队列

1. 独立进程，且队列数据可持久话，停止并重启队列后内存数据不丢失。
1. 支持一个public,多个subscripe模式，就是put一份数据，多个client去消费，所有client都get后，这份数据才在队列里消失。

内部开发完毕，备选redis。

### 自动化测试/每日构建/持续继承

不懂，目前只要求用unittest模块做单元测试。

### 相关工具

1. vim配置
    1. 语法，类库自动补全:python-dict, pyflake
    1. 语法检查，代码风格检查：flake8-vim
1. git使用规范
    1. 保持master分支稳定
    1. 把其它分支合并到master是用merge --no-ff，用git log --first-parent查看master分支要看到很干净的日志
1. 代码检查工具
    1. 代码复杂度检查：flake8
    1. 代码风格检查：flake8
    1. 检查重复代码：开发中
