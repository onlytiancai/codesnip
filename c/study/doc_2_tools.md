## 快速学习C语言二: 编译自动化, 静态分析, 单元测试，coredump调试，性能剖析

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

### 单元测试

安装CUnit

    wget http://sourceforge.net/projects/cunit/files/latest/download
    tar xf CUnit-2.1-3.tar.bz2
    cd CUnit-2.1-3
    ./bootstrap
    ./configure
    make
    make install

了解下单元测试的概念: 一次测试(registry)可以分成多个suit，一个suit里可以有多个
test case, 每个suit有个setup和teardown函数，分别在执行suit之前或之后调用。

下面的代码是一个单元测试的架子，这里测试的是库函数strlen，这里面只有一个suit,
就是testSuite1，testSuit1里里有一特test case，就是testcase，testcase里有一个
测试，就是test_string_length。

整体上就是这么一个架子，suit,test case, test都可以往里扩展。

    #include <assert.h> 
    #include <stdlib.h> 
    #include <string.h> 

    #include <CUnit/Basic.h>
    #include <CUnit/Console.h>
    #include <CUnit/CUnit.h>
    #include <CUnit/TestDB.h>

    // 测试库函数strlen功能是否正常
    void test_string_lenth(void){
        char* test = "Hello";
        int len = strlen(test);
        CU_ASSERT_EQUAL(len,5);
    }

    // 创建一特test case，里面可以有多个测试 
    CU_TestInfo testcase[] = {
        { "test_for_lenth:", test_string_lenth },
        CU_TEST_INFO_NULL
    };

    // suite初始化,
    int suite_success_init(void) {
        return 0;
    }

    // suite 清理
    int suite_success_clean(void) {
        return 0;
    }

    // 定义suite集, 里面可以加多个suit
    CU_SuiteInfo suites[] = {
        // 以前的版本没有那两个NULL参数，新版需要加上，否则就coredump
        //{"testSuite1", suite_success_init, suite_success_clean, testcase },
        {"testSuite1", suite_success_init, suite_success_clean, NULL, NULL, testcase },
        CU_SUITE_INFO_NULL
    };

    // 添加测试集, 固定套路
    void AddTests(){
        assert(NULL != CU_get_registry());
        assert(!CU_is_test_running());

        if(CUE_SUCCESS != CU_register_suites(suites)){
            exit(EXIT_FAILURE);
        }
    }

    int RunTest(){
        if(CU_initialize_registry()){
            fprintf(stderr, " Initialization of Test Registry failed. ");
            exit(EXIT_FAILURE);
        }else{
            AddTests();
            
            // 第一种：直接输出测试结果
            CU_basic_set_mode(CU_BRM_VERBOSE);
            CU_basic_run_tests();

            // 第二种：交互式的输出测试结果
            // CU_console_run_tests();

            // 第三种：自动生成xml,xlst等文件
            //CU_set_output_filename("TestMax");
            //CU_list_tests_to_file();
            //CU_automated_run_tests();

            CU_cleanup_registry();

            return CU_get_error();

        }

    }

    int main(int argc, char* argv[]) {
        return  RunTest();
    }

然后Makefile里增加如下代码

    INC=-I /usr/local/include/CUnit
    LIB=-L /usr/local/lib/

    test: testcase.c
        gcc -o test.o $(INC) $(LIB) -g  $^ -l cunit
        ./test.o

    .PHONY: test

再执行make test就可以执行单元测试了，结果大约如下

    gcc -o test.o -I /usr/local/include/CUnit -L /usr/local/lib/ -g  testcase.c -l cunit
    ./test.o


         CUnit - A unit testing framework for C - Version 2.1-3
         http://cunit.sourceforge.net/


    Suite: testSuite1
      Test: test_for_lenth: ...passed

    Run Summary:    Type  Total    Ran Passed Failed Inactive
                  suites      1      1    n/a      0        0
                   tests      1      1      1      0        0
                 asserts      1      1      1      0      n/a

    Elapsed time =    0.000 seconds

可以看到testSuite1下面的test_for_lenth通过测试了。
注意一下，安装完新的动态库后记得ldconfig，否则-l cunit可能会报错
如果还是不行就要 /etc/ld.so.conf 看看有没有 /usr/local/lib , 
cunit默认把库都放这里了。

### 调试coredump

就上面的单元测试, 如果使用注释掉那行，执行make test时就会产生coredump。如下

    // 定义suite集, 里面可以加多个suit
    CU_SuiteInfo suites[] = {
        {"testSuite1", suite_success_init, suite_success_clean, testcase },
        //{"testSuite1", suite_success_init, suite_success_clean, NULL, NULL, testcase },
        CU_SUITE_INFO_NULL
    };

但默认coredump不会保存在磁盘上，需要执`ulimit -c unlimited`才可以，然后要
指定一下coredump的路径和格式：

    echo "/tmp/core-%e-%p" > /proc/sys/kernel/core_pattern

其中%e是可执行文件名，%p是进程id。然后编译这段代码的时候要加上-g的选项，意思
是编译出调试版本的可执行文件，在调试的时候可以看到行号。

    gcc -o test.o -I /usr/local/include/CUnit -L /usr/local/lib/ -g  testcase.c -l cunit

在执行./test.o后就会产生一个coredump了，比如是/tmp/core-test.o-16793, 这时候
用gdb去调试该coredump，第一个参数是可执行文件，第二个参数是coredump文件

    gdb test.o /tmp/core-test.o-16793

挂上去后默认会有一些输出，其中有如下

    Program terminated with signal 11, Segmentation fault.

说明程序遇到了段错误，崩溃了，一般段错误都是因为内存访问引起的, 我们想知道
引起错误的调用栈， 输入bt回车，会看到类似如下的显示

    (gdb) bt
    #0  0x00007fe1b0b22cb2 in CU_register_nsuites () from /usr/local/lib/libcunit.so.1
    #1  0x00007fe1b0b22d28 in CU_register_suites () from /usr/local/lib/libcunit.so.1
    #2  0x0000000000400a8a in AddTests () at testcase.c:46
    #3  0x0000000000400adf in RunTest () at testcase.c:56
    #4  0x0000000000400b13 in main (argc=1, argv=0x7fff4fa51928) at testcase.c:79

这样大概知道是咋回事了，报错在testcase.c的46行上，再往里就是cunit的调用栈了，
我们看不到行号，好像得有那个so的调试信息才可以，目前还不会在gdb里动态挂符号文件
，所以就先不管了，输入q退出调试器，其它命令用输入help学习下。

    if(CUE_SUCCESS != CU_register_suites(suites)){

就调用了一个CU_register_suites函数，函数本身应该没有错误，可能是传给他从参数
有问题，就是那个suites，该参数构建的代码如下：

    CU_SuiteInfo suites[] = {
        {"testSuite1", suite_success_init, suite_success_clean, testcase },
        CU_SUITE_INFO_NULL
    };

是个CU_SuiteInfo的数组，就感觉是构建这个类型没构建对，然后就看他在哪儿定义
的

    # grep -n "CU_SuiteInfo" /usr/local/include/CUnit/*
    /usr/local/include/CUnit/TestDB.h:696:typedef struct CU_SuiteInfo {

在/usr/local/include/CUnit/TestDB.h的696行，具体如下

    typedef struct CU_SuiteInfo {
        const char       *pName;         /**< Suite name. */
        CU_InitializeFunc pInitFunc;     /**< Suite initialization function. */
        CU_CleanupFunc    pCleanupFunc;  /**< Suite cleanup function */
        CU_SetUpFunc      pSetUpFunc;    /**< Pointer to the test SetUp function. */
        CU_TearDownFunc   pTearDownFunc; /**< Pointer to the test TearDown function. */
        CU_TestInfo      *pTests;        /**< Test case array - must be NULL terminated. */
    } CU_SuiteInfo;

可以看到，该结构有6个成员，但我们定义的时候只有4个成员，没有设置pSetUpFunc和
pTearDownFunc的，所以做如下修改就能修复该问题了。

    -    {"testSuite1", suite_success_init, suite_success_clean, testcase },
    +    {"testSuite1", suite_success_init, suite_success_clean, NULL, NULL, testcase },

对了，gdb用yum安装就行了。

### 性能剖析

好些时候我们要去分析一个程序的性能，比如哪个函数调用了多少次，被谁调用了，
平均每次调用花费多少时间等。这时候要用gprof,gprof是分析profile输出的。
要想执行时输出profile文件编译时要加-pg选项，

    gcc -o helloworld.o -pg -g helloworld.c
    ./helloworld.o

执行上面语句后会在当前目录下生成gmon.out文件, 然后用gprof去读取并显示出来，
因为可能显示的比较长，所以可以先重定向到一个文件prof_info.txt里

        gprof -b -A -p -q helloworld.o gmon.out >prof_info.txt 

参数的含义先这么用，具体可以搜，最后查看prof_info.txt里会有需要的信息, 大概
能看懂，具体可以搜。

    Flat profile:

    Each sample counts as 0.01 seconds.
     no time accumulated

      %   cumulative   self              self     total           
     time   seconds   seconds    calls  Ts/call  Ts/call  name    
      0.00      0.00     0.00       15     0.00     0.00  cmp_default
      0.00      0.00     0.00       15     0.00     0.00  cmp_reverse
      0.00      0.00     0.00        4     0.00     0.00  w_strlen
      0.00      0.00     0.00        2     0.00     0.00  sort
      0.00      0.00     0.00        1     0.00     0.00  change_str_test
      0.00      0.00     0.00        1     0.00     0.00  concat_test
      0.00      0.00     0.00        1     0.00     0.00  customer_manager
      0.00      0.00     0.00        1     0.00     0.00  hello_world
      0.00      0.00     0.00        1     0.00     0.00  n_hello_world
      0.00      0.00     0.00        1     0.00     0.00  reverse
      0.00      0.00     0.00        1     0.00     0.00  sort_test
    
                Call graph


    granularity: each sample hit covers 2 byte(s) no time propagated

    index % time    self  children    called     name
                    0.00    0.00      15/15          sort [4]
    [1]      0.0    0.00    0.00      15         cmp_default [1]
    -----------------------------------------------
                    0.00    0.00      15/15          sort [4]
    [2]      0.0    0.00    0.00      15         cmp_reverse [2]
    -----------------------------------------------
                    0.00    0.00       1/4           reverse [10]
                    0.00    0.00       1/4           main [16]
                    0.00    0.00       2/4           concat_test [6]
    [3]      0.0    0.00    0.00       4         w_strlen [3]
    -----------------------------------------------

