## 快速学习C语言三: 开发环境, VIM配置, TCP基础，Linux开发基础，Socket开发基础

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

TCP使用很广泛，先了解一下概念，TCP是面向连接的协议，所以有建立连接和关闭连接
的过程。

建立连接过程需要三步握手，如下：

1. A向B发送syn信令
2. B向A回复ack，以及发送sync信令
3. A向B回复ack

其实网络上发送数据都有可能丢的，所以每个发送给对端的数据，要收到答复才能确认
对方收到了。
比如上面第二步A收到了B返回的ack才能确认连接已经建立成功，自己给B发送数据，B
可以收到，同样第三步B收到A的ack才能确认连接建立成功，自己发给A的数据，A能收到。
所以TCP连接建立不是两步握手，不是四步握手，而是三步握手。

连接建立成功后双方就可以互发psh信令来传输数据了，同样发出去的psh数据，也需要
收到ack才能确认对方收到，否则就得等待超时后重发。

拆除连接需要四步握手, 因为TCP是双工的，所以自己这边关闭连接，有可能对方还会
给自己发数据，还得等对方说自己不会给自己发送数据了。

1. A向B发送fin, 表示自己没有数据向B发送了。
2. B向A回复ack
3. B向A发送过fin, 表示自己没有数据向A发送了。
4. A向B回复ack

另外就是在任何时候都可能收到对方发来的rst信令，表示直接复位该连接，也别发数据了
也别等着收数据了，赶紧把资源都回收了吧。

TCP还有滑动窗口的流量控制机制，以及各种超时处理逻辑，有兴趣的话具体细节看
《TCP/IP协议详解》了。

linux下用tcpdump可以抓包学习TCP协议，比如在执行`curl -I www.baidu.com`时用
tcpdump抓包如下。

    # tcpdump -nn -t host www.baidu.com
    tcpdump: verbose output suppressed, use -v or -vv for full protocol decode
    listening on eth0, link-type EN10MB (Ethernet), capture size 65535 bytes
    IP 10.190.176.177.34840 > 180.97.33.71.80: Flags [S], seq 1772495094, win 14600, options [mss 1460,sackOK,TS val 214360452 ecr 0,nop,wscale 5], length 0
    IP 180.97.33.71.80 > 10.190.176.177.34840: Flags [S.], seq 946873815, ack 1772495095, win 14600, options [mss 1440,sackOK,nop,nop,nop,nop,nop,nop,nop,nop,nop,nop,nop,wscale 7], length 0
    IP 10.190.176.177.34840 > 180.97.33.71.80: Flags [.], ack 1, win 457, length 0
    IP 10.190.176.177.34840 > 180.97.33.71.80: Flags [P.], seq 1:168, ack 1, win 457, length 167
    IP 180.97.33.71.80 > 10.190.176.177.34840: Flags [.], ack 168, win 202, length 0
    IP 180.97.33.71.80 > 10.190.176.177.34840: Flags [P.], seq 1:705, ack 168, win 202, length 704
    IP 10.190.176.177.34840 > 180.97.33.71.80: Flags [.], ack 705, win 501, length 0
    IP 10.190.176.177.34840 > 180.97.33.71.80: Flags [F.], seq 168, ack 705, win 501, length 0
    IP 180.97.33.71.80 > 10.190.176.177.34840: Flags [.], ack 169, win 202, length 0
    IP 180.97.33.71.80 > 10.190.176.177.34840: Flags [F.], seq 705, ack 169, win 202, length 0
    IP 10.190.176.177.34840 > 180.97.33.71.80: Flags [.], ack 706, win 501, length 0

可以看到本机的ip是10.190.176.177，baidu解析出来的ip是180.97.33.71，然后前三个
包就是建立连接的三步握手，最后三个包是关闭连接的四步握手。中括号里的S表示sync,
p表示psh，F表示fin，.好像表示ack。


### Linux基础

其实Linux下，C的库函数，以及linux API都在libc.so里面，没有分开
的。玩C语言开发，肯定要对C库函数和常用的linux API有所熟悉的，可以先看
如下两个链接快速了解一下，知道系统有哪些能力和轮子。

Standard C 语言标准函数库速查
http://ganquan.info/standard-c/
Linux系统调用列表
http://www.ibm.com/developerworks/cn/linux/kernel/syscall/part1/appendix.html

再就是系统调用，Linux API, 系统命令，和内核函数不是一回事，虽然他们有关联。
系统调用是通过软中断向内核提交请求，获取内核服务的接口，Linux Api则定义了一组
函数如read,malloc等，封装了系统调用, 比如malloc函数会调用brk系统调用。
然后有系统命令则更高一级，如ls，hostname,则直接提供了一个可执行程序, 关于他们
的关系可以阅读下面这篇文章：

http://wenku.baidu.com/view/9e33f3e94afe04a1b071de81.html

C语言要想使用别人的东西，首先要包含别人提供的头文件，使用linux api和c库函数
也一样，默认的这些头文件都在/usr/include里，自己安装的一些则一般约定放在
/usr/local/include里。写代码的过程中如果遇到一些类型或函数不知道怎么使用，直接
可以在这里面找到头文件看源码。

Linux下还有好多数据类型是在学普通C语言是没见到过的，比如size_t,ssize_t,unit32_t
啥的, 这些其实都在普通数据类型的别名，一般在/usr/include/asm/types.h里可以看到
他们是怎么被typedef的，使用这些类型主要是为了提高可移植性，同时语义更加明确，
比如size_t在32位机器上定义为uint，64位机器上定义为ulong，使用size_t编写的代码
就可以在32位机器和64位机器上良好运行。 还有size_t的意义更明确，它不是用来表示
普通的无符号数字概念的，而是表示sizeof返回的结果或者说是能访问的体系内存的长度。

然后像uint32_t这种类型是为了编写出更明确的代码，像C语言的类型，int, long等在
不同的机器上都有不同的长度，但uint32_t在啥机器上都是32位长的，有时候需求就是
这样，就需要用这种数据类型了。

还有就是Linux系统函数调用失败，大多数时候都会erron赋一个整数值，这个整数值可以
表示不同的错误原因，可以在终端下运行man errno来查看详细，另外好多系统函数都可以
用man来查看帮助的，有的里面还有使用示例的，是学习linux编程的很好的工具。

还有一些系统函数设计的挺好，我总结了一些惯用法吧算是，自己设计函数也可以学习

第一个是通过指针参数来获取数据，因为好多函数的返回值是int类型，表示函数调用
是否成功，或错误码，而这个函数本身的任务还要返回一些实质的信息，这时候就可以
通过参数来填充数据，让调用者拿到，比如accept函数的使用
(简化后的伪代码，不能执行):

    struct sockaddr_in client;
    if (accept(listenfd, &client) >= 0) {
        printf("%s\n", client);
    }

这样我们调用一次函数，既能知道有没有调用成功，成功的话又能拿到客户端的描述符,
以及对端的网络地址。

第二个是C没有类和对象的概念，但也可以模拟出来类似的概念，比如网络编程，通过
socket函数创建一个描述符，比如说是fd，其实这就相当于一个类的实例，一个对象了，
然后调用read(fd),send(fd),close(fd)等函数来操作它，和面向对象里用fd.read(),
fd.send(),fd.close()只是用法不同而已，所以写C是能用得到一些面向对象的思想的。

第三个是在Linux里好多东西可以用描述符来表示，比如文件，硬件端口，网络连接等，
然后可以针对描述符调用read,write等操作，这个是个很好的抽象，可以使用很简单的
几个接口来实现很强大的功能，在写自己的C软件时也可以借鉴这个思路。就是先建立一个
概念，然后写很多的函数来操作这个概念，而不是建立很多的概念，大家记不住的。

第四个是，C其实没有太多的类型检查功能，表示复杂的数据都用struct表示，而不同的
struct是可以强转的，所以可以用带标志的struct来表达类似面向对象多态的概念，如
bind函数需要一个struct sockaddr的参数，但ipv4和ipv6的地址分别用
struct sockaddr_in和struct sockaddr_in6表示，感觉就相当于struct sockaddr的两个
子结构，这样bind函数就使用父结构struct sockaddr来同时支持ipv4和ipv6了。
需要注意子结构和父结构的标志成员要放在最前面，这样子结构转成父结构时，父结构
才能正确的读出标志，从而在具体使用时强转为合适的子结构。

就这样了，Linux编程入门我知道的就这些，更多可看《Unix环境高级编程》

### socket基础

先学一些socket客户端编程来熟悉socket编程吧, 要连接到远程主机，首要要
有个远程主机的地址，一个远程主机的地址包含对方的IP和端口，有时候我们
只知道对方的域名，所以首先要解析出IP来，好多书上都是用gethostbyname来解析域名
的，但它过时了，不支持ipv6，而且参数不支持ip格式的字符串，返回的地址必须拷贝
后才能使用，否则同线程再调用一次该函数那地址就变了，总之是一个过时的函数了。

现在比较国际范的函数是getaddrinfo，可以通过man查它的用法，

    int getaddrinfo(const char *node, const char *service,
                    const struct addrinfo *hints,
                    struct addrinfo **res);

该函数同时支持 ipv4和v6，然后host支持域名也支持ip格式的字符串，hints用来设置
查询的一些条件，result用来获取查询到的结果，他是一个指向指针的指针类型。

这相当也是一个惯用法了，一个参数用来说明调用需求，一个指针参数来获取返回数据。
像select就是调用需求和返回数据都是一个参数来表示，但像pool就是调用需求和返回
用两个参数了，更明确，前一个是const，后一个是指针。具体使用示例如下：

    struct addrinfo* get_addr(const char *host, const char *port){
        struct addrinfo hints;     // 填充getaddrinfo参数
        struct addrinfo *result;   // 存放getaddrinfo返回数据

        memset(&hints, 0, sizeof(struct addrinfo));
        hints.ai_family = AF_UNSPEC;
        hints.ai_socktype = SOCK_STREAM;
        hints.ai_flags = 0;
        hints.ai_protocol = 0;

        if(getaddrinfo(host, port, &hints, &result) != 0) {
            printf("getaddrinfo error");
            exit(1);
        }
        return result;
    }

对了，getaddrinfo返回的result指向的内存是系统分配的，用完了要调用
freeaddrinfo去释放内存的。其实getaddrinfo的内部实现挺复杂的，调用了一堆ga开头
的函数，而且struct addrinfo其实也蛮复杂的，里面有好多信息，但用好它是写出
同时支持ipv4,ipv6网络程序的关键。

创建socket, 要熟悉下family,socktype,protocol等概念和取值，查man吧

    int create_socket(const struct addrinfo * result) {
        int fd;

        if ((fd = socket(result->ai_family, result->ai_socktype, result->ai_protocol)) == -1) {
            printf("create socket error:%d\n", fd);
            exit(-1);
        }
        printf("cerate socket ok: %d\n", fd);
        return fd;
    }

连接目标主机, 这里其实就是要三步握手了，有几个常见的错误，可以通过检测errno来
读取，如ETIMEDOUT表示建立连接超时，就是发出去sync没人搭理，或ECONNREFUSED表示
对方端口没开，发过去的sync直接被对方发了个rst回来，或EHOSTUNREACH表示对方机器
没开或宕机了，因为ICMP包返回错误了。

    int connect_host(int fd, const struct addrinfo* addr) {
        if (connect(fd , addr->ai_addr, addr->ai_addrlen) == -1) {
            printf("connect error.\n");
            exit(-1);
        }
        printf("collect ok\n");
        return 0;
    }

我们要做一个HTTP客户端，类似curl，要拼一个HTTP请求发送给远程主机，拼包用
snprintf虽然弱了一点，但也是最容易理解的，先用着。要留意格式化后的字符串大小
别超过缓冲区大小，当然了指定了长度不会溢出，但超过后会截断，如果HTTP请求丢失
了最后的两对\r\n，服务端就不知道客户端发送完数据了, 所以这里边界处理要十分
小心，可能我这里写的还有BUG。

还有就是数据大的话send一次可能发送不完，这里先简单粗暴处理了一下，真实程序的
话要把剩下的半拉重新拷贝个buf发出去的。

    int get_send_data(char * buf, size_t buf_size, const char* host) {
        const char *send_tpl;                        // 数据模板，%s是host占位符 
        size_t to_send_size;                         // 要发送到数据大小 

        send_tpl = "GET / HTTP/1.1\r\n"
                   "Host: %s\r\n"
                   "Accept: */*\r\n"
                   "\r\n\r\n";

        // 格式化后的长度必须小于buf的大小，因为snprintf会在最后填个'\0'
        if (strlen(host) + strlen(send_tpl) - 2 >= buf_size) { // 2 = strlen("%s")
            printf("host too long.\n");
            exit(-1);
        }

        to_send_size = snprintf(buf, buf_size, send_tpl, host);
        if (to_send_size < 0) {
            printf("snprintf error:%s.\n", to_send_size);
            exit(-2);
        }

        return to_send_size;
    }

    int send_data(int fd, const char *data, size_t size) {
        size_t sent_size;
        printf("will send:\n%s", data);
        sent_size = write(fd, data, size);
        if (sent_size < 0) {
            printf("send data error.\n");
            exit(-1);
        }else if(sent_size != size){
             printf("not all send.\n");
             exit(-2);
        }
        printf("send data ok.\n");
        return sent_size;
    }

完了收数据，我们只取HTTP应答第一行就好了，然后关闭连接。协议解析也简单粗暴
找到\r\n就停止，真实程序可能要写个状态机来解析了。

    int recv_data(int fd, char* buf, int size) {
        int i;
        int recv_size = read(fd, buf, size);
        if (recv_size < 0) {
            printf("recv data error:%d\n", (int)recv_size);
            exit(-1);
        }
        if (recv_size == 0) {
            printf("recv 0 size data.\n");
            exit(-2);
        }
        // 只取HTTP first line
        for (i = 0; i < size - 1; i++) {
            if (buf[i] == '\r' && buf[i+1] == '\n') {
                buf[i] = '\0';
            }
        }
        printf("recv data:%s\n", buf);
    }

    int close_socket(int fd) {
        if(close(fd) < 0){
             printf("close socket errors\n");
             exit(-1);
        }
        printf("close socket ok\n");
    }

最后用main函数把他们串起来

    int main(int argc, const char *argv[])
    {
        const char* host = argv[1];                  // 目标主机
        char send_buff[SEND_BUF_SIZE];               // 发送缓冲区
        char recv_buf[RECV_BUFF_SIZE];               // 接收缓冲区
        size_t to_send_size = 0;                     // 要发送数据大小 
        int client_fd;                               // 客户端socket
        struct addrinfo *addr;                       // 存放getaddrinfo返回数据

        if (argc != 2) {
            printf("Usage:%s [host]\n", argv[0]);
            return 1;
        }


        addr = get_addr(host, "80");
        client_fd = create_socket(addr);
        connect_host(client_fd, addr);
        freeaddrinfo(addr);

        to_send_size = get_send_data(send_buff, SEND_BUF_SIZE, host);
        send_data(client_fd, send_buff, to_send_size);

        recv_data(client_fd, recv_buf, RECV_BUFF_SIZE);

        close(client_fd);
        return 0;
    }

### 小结

多看，多写，多练，肯定能熟悉C语言的，我现在看好多C的书都能看懂了。
