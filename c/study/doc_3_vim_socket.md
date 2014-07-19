## 快速学习C语言三: 开发环境和Socket客户端编程 

上次学了一些C开发相关的工具，这次再配置一下VIM，让开发过程更爽一些。
另外再学一些linux下网络开发的基础，好多人学C也是为了做网络开发。

### 开发环境

首先得有个Linux环境，有时候家里机器是Windows，装虚拟机也麻烦，所以还不如30块钱
买个腾讯云，用putty远程练上去写代码呢。

我一直都是putty+VIM在Linux下开发代码，好几年了，只要把putty和VIM配置好，其实
开发效率挺高的。

买好腾讯云后，装个Centos，会分配个外网IP，然后买个域名，在DNSPod解析过去，就
可以用putty远程登录了，putty一般做如下设置。

- window\appearance\font setting:consolas 12pt ， 设置字体
- window\translate\charset:utf-8 , 设置字符集
- window\selection\action of mouse buttons:windows .. , 设置可以用鼠标选择文字
- window\line of scoreback:20000 ，设置可滚屏的长度
- connection\auto-login username:root, 设置自动登录的用户名
- connection\seconds of keepalive:10, 设置心跳，防止自动断开

设置完成后把这个会话起个名字，比如叫qcloud，下次用的时候先加载，然后open
就可以了, 所有设置会保存起来。这样配置后putty已经很好用了，但我们还可以搞成
自动登录，不需要每次都输入密码。

- 在Linux下ssh-keygen -t rsa 生成密钥对
- 把私钥id_isa下载到用scp下载到windows并用puttygen加载并重新保存私钥。
- 在windows下新建快捷方式输入D:\soft\putty.exe -i D:\ssh\wawa.ppk -load 
  "qcloud" 其中-i 指定私钥位置，-load指定会话名称，

下次双击快捷方式就登录上去了，而且上面的设置都会生效。对了，putty和puttygen
要在官方下载哦。

### VIM配置

首先安装最新的VIM. 

    wget ftp://ftp.vim.org/pub/vim/unix/vim-7.4.tar.bz2
    ./configure --prefix=/usr/local/vim --enable-multibyte --enable-pythoninterp=yes
    make
    make install

修改下~/.bashrc, 加入如下两句，可以让vim和vi指定成刚安装的版本

    alias vim='/usr/local/vim/bin/vim'
    alias vi='vim'

简单配置下VIM，就可以开工了, 打开~/.vimrc，添加如下:

    " 基本设置
    set nocp
    set ts=4
    set sw=4
    set smarttab
    set et
    set ambiwidth=double
    set nu

    " 编码设置 
    set encoding=UTF-8
    set langmenu=zh_CN.UTF-8
    language message zh_CN.UTF-8
    set fileencodings=ucs-bom,utf-8,cp936,gb18030,big5,euc-jp,euc-kr,latin1
    set fileencoding=utf-8

基本每一个VIM最少要配置成这样，包括生产环境，前半拉主要是设置缩进成4个空格，
后半拉是设置编码，以便打开文件时不会乱码。

如果想开发时更爽一些，就得装插件了，现在装插件也很简单，先装插件管理工具
pathogen.vim, 如下

    mkdir -p ~/.vim/autoload ~/.vim/bundle && curl -LSso ~/.vim/autoload/pathogen.vim https://tpo.pe/pathogen.vim

然后安装一个文件模版插件，一个代码片段插件，一个智能体似乎插件就可以了，傻瓜式
的，如下
    
    # 安装vim-template
    cd ~/.vim/bundle
    git clone git://github.com/aperezdc/vim-template.git

    # 安装snipmate
    git clone git://github.com/msanders/snipmate.vim.git
    cd snipmate.vim
    cp -R * ~/.vim

    # 安装clang_complete
    yum install clang
    git clone https://github.com/Rip-Rip/clang_complete.git
    cd clang_complete/
    make install

再在~/.vimrc里加入如下两句

    execute pathogen#infect()
    syntax on
    filetype plugin indent on

别的插件能不装就不装了吧，用的时候再说，现在你打开一个新的.c文件，会自动从模版
里加载一个代码框架进来，然后输入main,for,pr等按tab键就会自动生成代码片段，
然后include头文件后，里面的函数，类型等在输入时按ctrl+n就会自动提示，结构的
成员也可以，已经很爽了。

### TCP基础

建立连接，关闭连接，滑动窗口

### linux基础

库函数，libc里，
头文件位置
size_t, ssize_t
文件描述符, read, write
系统调用
errno

### socket基础

socket, connect, gethostbyname, getaddrinfo, send, recv
