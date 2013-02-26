安装了gcc-objc,GNUstep，写了第一个Objective-C程序并能正常运行

使用

    make
    ./HelloWorld.app/HelloWorld

参考链接

1. [CentOS下安装gnustep, gorm and project center](http://t.cn/zYWymWY)
1. [Compile Objective-C Programs Using gcc](http://t.cn/GAJxp)
1. [GNUstep 入門](http://t.cn/zYWq0Aj)

bashrc需要加以下几行

    GNUSTEP_MAKEFILES=/usr/GNUstep/System/Library/Makefiles
    export GNUSTEP_MAKEFILES
    . /usr/GNUstep/System/Library/Makefiles/GNUstep.sh
