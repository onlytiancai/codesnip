# 如何编写高质量的python程序

## 目录

1. 代码规范
1. 空白项目模版
1. 单元测试
1. 文档
1. 打包
1. 小结

## 代码规范

首先阅读下面的两份规范，并深入理解。

- [Python社区官方建议采用的Python编码风格:PEP8](http://www.python.org/dev/peps/pep-0008/) [中文版](http://wiki.woodpecker.org.cn/moin/PythonCodingRule)
- [Google SoC 建议的 Python 编码风格:Google Python Style Guide](http://google-styleguide.googlecode.com/svn/trunk/pyguide.html) [中文版](http://www.elias.cn/Python/PythonStyleGuide)

写出规范的代码是写出高质量代码的第一步，并且有助于培养仔细的习惯。

为了培养规范写代码的习惯，可以安装[flake8](http://pypi.python.org/pypi/flake8/)这个工具，它不仅可以检查代码风格是否符合官方建议（PEP8），而且还能找出潜在的隐患（用Pyflakes做语法分析），更逆天的是还能检测到你有些函数写的太复杂（代码圈复杂度）了，更更逆天的是可以设置git commit之前必须通过这些检查。

当然具体操作需要根据自己的项目进行一些定制，比如可以忽略E501，W293。

## 空白项目模版

好的开始是成功的一半，写python代码就从[pyempty](https://github.com/onlytiancai/pyempty)开始吧。

在github上看一下那些经典的项目，[web.py](https://github.com/webpy/webpy),[flask](https://github.com/mitsuhiko/flask), [pep8](https://github.com/jcrocholl/pep8/blob/master/pep8.py)，他们的项目目录都很规范，综合借鉴了一些项目的特点，我写了这个pyempty项目。

1. **README.md** 这里写你项目的简介，quick start等信息，虽然distutils要求这个文件没有后缀名，但github上如果后缀是.md的话可以直接转换成html显示。
1. **ChangeLog.txt** 该文件存放程序各版本的变更信息，也有一定的格式，参考[web.py的ChangeLog.txt](https://github.com/webpy/webpy/blob/master/ChangeLog.txt)
1. **LICENES.txt** 这里存放你项目使用的协议，不要编写自己的协议。
1. **requirements.txt** 如果你的项目需要依赖其它的python第三方库，在这里一行一个写出来，可能pip install的时候能自动帮你安装
1. **setup.py** 安装脚本，后面详细介绍
1. **docs** 里面存放你的项目文档，如概要设计，详细设计，维护文档，pydoc自动生成的文档等，强烈推荐大家使用MarkDown格式编写文档
1. **src** 这个目录里存放项目模块的主要代码，尽量不要把模块目录直接放到根目录，模块代码目录可以在setup.py里指定的
1. **tests** 这个目录存放所有单元测试，性能测试脚本，单元测试的文件确保以test_做前缀，这样distutils会自动打包这些文件，并且用`python -m unittest discover -s ./ -p 'test_*.py' -v` 可以直接执行这些测试

## 单元测试

    Martin Fowler："在你不知道如何测试代码之前，就不该编写程序。而一旦你完成了程序，测试代码也应该完成。除非测试成功，你不能认为你编写出了可以工作的程序。"

我们有很多理由不写单元测试，归根结底是懒，虽然[代码大全上说](http://www.cnblogs.com/onlytiancai/archive/2010/05/26/1744108.html)：

    大部分研究都发现，检测比测试的成本更小。NASA软件工程实验室的一项研究发现，阅读代码每小时能够检测出来的缺陷要比测试高出80%左右(Basili and Selby 1987)。后来，IBM的一项研究又发现，检查发现的一个错误只需要3.5个工作时，而测试则需要花费15-25个工作时（Kaplan 1995)。

但是单元测试还是让别人相信你的代码有很高质量的最有力证据。

好了，请详细阅读：

1. [深入python3.0: 单元测试-2.x也适用](http://woodpecker.org.cn/diveintopython3/unit-testing.html)
1. [Unit testing framework](http://docs.python.org/library/unittest.html) [不完整中文版](http://www.ibm.com/developerworks/cn/linux/l-pyunit/index.html)

## 文档

敏捷开发不是提倡什么文档也不写，没有文档就没有传承和积累，轮岗或新人接手任务就会遇到很大的麻烦，所以我决定每个项目最少要写以下文档：

1. **nalysis.model.md** 概要设计文档，不同于README.md文件，该文档应该写于项目开发之前，把项目有哪些功能，大概分几个模块等项目整体概述信息写一下。
1. **design.model.md** 详细设计文档，不用太详细，至少把项目依赖哪些东西，谁依赖这个项目，重要算法流程描述，代码整体结构等写出来。
1. **maintain.md** 维护文档，这个我觉得最重要，你的服务都记录哪些日志，需要监控哪些业务指标，如何重启，有哪些配置项等，没这些东西，你的项目很难运维。

上面这些文档都是项目全局性的文档，不适合写在docstring或注视里，所以要有单独的文档。

## 打包

python有专门的模块打包系统[distutils](http://docs.python.org/library/distutils.html)，你可以用这套机制把你的代码打包并分发到[Pypi](http://pypi.python.org/pypi)上，这样任何人都可以用[pip](http://pypi.python.org/pypi/pip)或[easy_install](http://pypi.python.org/pypi/setuptools)安装你的模块。

如果你开发的是内部项目，还可以用[mypypi](http://pypi.python.org/pypi/mypypi)架设私有的pypi，然后把项目的大的版本更新发布到内部的pypi上，配置管理人员和运维人员可以很方便的从pypi上拉取代码安装到测试环境或生产环境。

发布大版本的时候要给版本命名及编写ChangeList，可以参考[Git Pro的相关章节](https://github.com/chunzi/progit/blob/master/zh/05-distributed-git/01-chapter5.markdown),主要记住以下几个命令。

    git tag -a v0.1 -m 'my test tag'  #给大版本命名，打Tag
    git describe master #给小版本命名,Git将会返回一个字符串，由三部分组成：最近一次标定的版本号，加上自那次标定之后的提交次数，再加上一段SHA-1值
    git shortlog --no-merges master --not v0.1 #生成版本简报,ChangeList

python有自己的打包机制，所以一般不要用`git archive`命令。

当然大版本管理用pypi管理比较合适，小的bug fix，紧急上线等好多公司都是用git直接从生产环境拉代码更新，因为git,svn等可以很方便的撤销某次更新，回滚到某个位置。

如何管理好大版本上线和小的紧急上线，我还没理清思路，欢迎大家参与讨论。

关于打包，请阅读如下链接：

1. [Python 打包指南](http://www.ibm.com/developerworks/cn/opensource/os-pythonpackaging/)
1. [深入Python3.0:打包 Python 类库](http://woodpecker.org.cn/diveintopython3/packaging.html)
1. [python打包:分发指定文件](http://docs.python.org/release/3.1.5/distutils/sourcedist.html#manifest)

## 小结

以上是最近学到的一些东西的总结，欢迎大家一起讨论。
