## 快速学习C语言二: 编译自动化, 静态分析, 单元测试，性能剖析，内存分析，日志记录 

上次的Hello world算是入门了，现在学习一些相关工具的使用

### 编译自动化 

写好程序，首先要编译，就用gcc就好了，基本用法如下

	gcc helloworld.c -o helloworld.o

helloworld.c是源码，helloworld.o是编译后的可执行文件，运行的话就用
`./helloworld.o`就可以了。

但是如果代码写的多了，每次改动完都手动用gcc编译太麻烦了，所以要用Makefile来
自动化这项工作，在当前目录下创建Makefile文件，大概如下

    helloworld.o: helloworld.c
        gcc helloworld.c -o helloworld.o

    .PHONY: lint 
    lint:
        splint helloworld.c -temptrans -mustfreefresh -usedef

    .PHONY: run
    run:
        ./helloworld.o

    .PHONY: clean
    clean:
        rm *.o 

缩进为0每一行表示一个任务，冒号左边的是目标文件名，冒号后面是生成该目标的依赖
文件，多个的话用逗号隔开，如果依赖文件没有更改，则不会执行该任务。

缩进为1的行表示任务具体执行的shell语句了，.PHONY修饰的目标表示不管依赖文件
有没有更改，都执行该任务。

执行对应的任务的话，就是在终端上输入`make 目标名`，如`make lint`表示源码检查，
`make clean`表示清理文件，如果只输入make，则执行第一个目标，对于上面的文件就
是生成helloworld.o了。

现在修改完源码，值需要输入一个make回车就行了，Makefile很强大，可以做很多自动化
的任务，甚至测试，部署，生成文档等都可以用Makefile来自动化，有点像前端的
Grunt和Java里的ant，这样就比较好理解了。


### 静态检查

静态检查可以帮你提前找出不少潜在问题来，经典的静态检查工具就是lint，具体到
Linux上就是splint了，可以用yum来安装上。

具体使用的话就是`splint helloworld.c`就行了，它会给出检查出来的警告和错误，还
提供了行号，让你能很快速的修复。

值得注意的是该工具不支持c99语法，所以写代码时需要注意一些地方，比如函数里声明
变量要放在函数的开始，不能就近声明，否则splint会报parse error。

静态检查工具最好不要忽略warning，但是有一些警告莫名其妙，我看不懂，所以还是
忽略了一些，在使用中我加上了`-temptrans -mustfreefresh -usedef`这几个参数。
