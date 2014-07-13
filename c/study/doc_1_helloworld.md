## 快速学习C语言一: Hello World

估计不会写C语言的同学也都听过C语言，从头开始快速学一下吧，以后肯定能用的上。
如果使用过其它类C的语言，如JAVA，C#等，学C的语法应该挺快的。

先快速学习并练习一些基本的语言要素，基本类型，表达式，函数，循环结构，
基本字符串操作， 基本指针操作，动态分配内存，使用结构表示复杂数据，
使用函数指针实现灵活逻辑。

虽然C是一个规模很小的语言，但也得自己多设计一些练习练手才能学会。

### 基本类型

我就记得char, int, 别的都不常用吧应该，用的时候再搜索。

### 表达式

和JAVA, C#差不多吧，不用学基本，各种算数运算符，关系运算符，逻辑运算符，逗号，
括号等的意思应该也差不多，表达式最终的结果也有类型和值。

### 函数

函数是最基本的抽象，基本没有什么语言没有函数的概念，它封装一系列操作,
最简单的Hello world，如下。

    static void hello_world(){
        printf("hello, world\n");
    }

我们的练习都是随手写的函数，不需要被外部调用，所以前面加个static，表示只在
本文件内可见。

`printf`输出一行的话，最后要加`\n`, 常见个格式化参数有%d,%c,%s,%p等，分别表示
输出int, char, 字符串, 指针。

### 分支，循环结构

和别的语言差不多，不过i的声明要放在函数开头，c89就是这样。

    static void n_hello_world(int n){
        int i = 0;
        for (i = 0; i < n; i++) {
            printf("hello, world\n");
        }
    }


### 字符串练习，获取一个字符串的长度

库函数`strlen`就是干这个的，不过我们自己可以写一个练手，c没有字符串类型，
用'\0'结尾的字符数组表示字符串，所以for循环从头滚到'\0'位置就好了。

    // 字符串练习, 计算字符串长度
    static int w_strlen(const char* str){
        int i;
        // 向后滚动指针，同时递增i，直到找到字符串结尾
        for (i = 0; *str != '\0'; str++, i++) {
            ;
        }
        return i;
    }

const 修饰符表示这个参数不能在函数里进行更改，防止意外改动。char *就是传说中
字符串了。
写C程序得用好for语句，有各种惯用法，用好了可以写出很紧凑的程序，比如上面for语句
的第2个分号后的逗号表达式可以递增两个变量。

### 理解字符串的存储

第一种方式是在编译时分配的内存，是字符串常量，指针s1指向的内存不能更改。
第二种方式应该是在栈上分配的内存(不确定)，可以通过指针修改其中的字符。

    static void change_str_test(){
        // 常量不能修改
        // char* s1 = "hello"; // will core dump
        char s1[10] = "hello";
        *s1 = 'p';
        printf("%s\n", s1);
    }


### 指针练习

指针可以进行加减的操作，每加一次就滚动过它指向的类型长度, 比如char指针就是
滚动1个字节。

    // 指针练习, 反转字符串
    static char* reverse(char* str){
        char* ret = str;
        // 滚到字符数组末尾的\0之前
        char* p = str + w_strlen(str) - 1;
        char c;
      
        // 两个指针，一个从前往后滚，一个从后往前滚，直到就要交错之前
        // 滚动的过程中交换两个指针指向的字符
        for ( ; p > str; --p, ++str) { 
            printf("debug[reverse]: %p %p %c %c\n", p, str, *p, *str);
            c = *p;
            *p = *str;
            *str = c;
        }

        return ret;
    }


`c = *p`表示取出指针p指向的字符，赋值给变量c，*表示取值。

`*p = *str`相当于`p[i] = str[i]`，右边的*取出来的是值，左边的*取出来的也是值,
值赋值给值，看起来有些诡异，但就是这样写的。反正`p = *str`肯定不对，因为p是
指针类型，*str是计算结果是字符类型。

### 动态分配内存

我记得TCPL前几章都没讲malloc,free等内存分配的函数，好多任务只需要在编译阶段
分配内存就够了，但比较大型复杂的程序应该都需要动态管理一些内存的。

C语言没有GC，要手工释放动态分配的内存，否则就会造成内存泄漏，所以一定要配平
资源，有malloc的地方，一定要想好它应该在哪里free。

目前我了解到的原则就有两种：

- 谁分配，谁释放
- 谁使用，谁释放

对了, malloc出来的内存要强转成你需要的指针类型，然后free时指针要滚到你动态
分配内存的起始点。

    // 内存申请相关，连接两个字符串
    static void concat_test(){
        char* a = "hello";
        char* b = "world";
        //结果字符串长度为两个字符窜长度加\0的位置
        int len = w_strlen(a) + w_strlen(b) + 1;
        // 动态分配内存
        char* p = (char *)malloc(sizeof(char) * len);
        char* result; 

        // 必须判断是否分配到内存
        if (p != NULL){
            // 保存动态分配内存的开始指针，free时必须从这里free
            result = p;

            //滚动p和a，直到a的末尾
            while (*a != '\0') {
                printf("debug[concat_test]:while a %p %c\n", a, *a);
                *p++ = *a++;
            }

            //滚动p和b，直到b的末尾
            while (*b != '\0') {
                printf("debug[concat_test]:while b %p %c\n", a, *a);
                *p++ = *b++;
            }

            // 末尾整个0
            *p= '\0';
            printf("concat_test: %s\n", result);

            //释放动态分配的内存
            free(result);
        }else{
            printf("malloc error"); 
        }
    }

### 结构练习

C没有类，要表达复杂的数据，就得用结构了, 结构也可以用指针来指，如果是结构变量
的话，引用成员用`.`，如果是指向结构的指针，引用成员用`->`

别的好像没啥特别的，注意动态分配结构数组后，指针滚动的边界，别使用了界外的
内存。如果结构的成员指向的内存是动态分配的花，也记得free。

没有结构，估计写不出大程序，结构应该会用的很多。

    //结构练习，人员统计系统
    struct customer {
        char* name;
        int age;
    };

    static void customer_manager() {
        // 直接在栈上分配结构体
        struct customer wawa;
        struct customer* p_wawa;
        struct customer* p_customers;
        int n = 2;

        char name[] = "wawa";
        // char* name = "wawa"; //splint warning
        char name2[] = "tiancai";
       
        // 直接用结构名访问成员 
        wawa.name = name;
        wawa.age = 30;
        printf("%s is %d years old\n", wawa.name, wawa.age);

        // 用指针访问结构成员
        p_wawa = &wawa;
        p_wawa->age = 31;
        printf("%s is %d years old\n", wawa.name, wawa.age);

        // 为员工数组动态分配内存
        p_customers = (struct customer*)malloc(sizeof(struct customer) * n);
        if (p_customers != NULL) {
            // 设置数组第一项
            p_customers->name = name;
            p_customers->age = 10;

            // 设置数组第二项
            p_customers++;
            p_customers->name = name2;
            p_customers->age = 30;

            // 滚动数组外面，然后反向循环到数组开始
            p_customers++;
            while(n-- > 0){
                p_customers--;
                printf("%s is %d years old\n", p_customers->name, p_customers->age);
            }

            // 释放动态分配的内存，这时候p_customers已经位于起始位置了
            // 结构体里的name1, name2是在栈上分配的，不用释放
            free(p_customers);
        }
    }

### 函数指针练习

好多语言都有高阶函数的特性，比如函数的参数或返回值还可以是个函数，
C里也有函数指针可以达到类似的效果，用来做回调函数等。

但C的函数指针写起来比较诡异，不好记忆，不行就用typedef来重新命个名，写起来
简单一些。

下面用一个比较经典的冒泡排序来演示函数指针的使用，传递不同的比较函数可以
改变排序函数的行为，这是写复杂灵活逻辑的一种很方便的方式。

    // 函数指针练习, 排序

    // 正序排序的比较函数
    static int cmp_default(int a, int b){
        return a - b;
    }

    // 反序排序的比较函数
    static int cmp_reverse(int a, int b){
        return b - a; 
    }

    // int类型的冒泡排序算法，可传入一个比较函数指针
    // 类似回调函数，该函数需要两个int参数且返回int
    static void sort(int* arr, int n, int (*cmp)(int, int)){
        int i, j, t;
        int *p, *q;

        p = arr;

        for (i = 0; i < n; i++, p++) {
            q = p;
            for (j = i; j < n; j++, q++) {
                // 调用函数指针指向的函数和使用函数一样，貌似是简单写法
                if (cmp(*p, *q) > 0) {
                    t = *p;
                    *p = *q;
                    *q = t;
                }
            }
        }
    }

    // 测试排序函数
    static void sort_test(){
        int arr[] = {4, 5, 3, 1, 2};
        int i, n = 5;

        // 正向排序， 传入cmp_default函数的地址，貌似不需要&取地址
        sort(arr, 5, cmp_default);
        for (i = 0; i < n; i ++) {
            printf("%d%s", arr[i], i == n - 1 ? "" : ", "); 
        }
        printf("\n");

        //反向排序，同上
        sort(arr, 5, cmp_reverse);
        for (i = 0; i < n; i ++) {
            printf("%d%s", arr[i], i == n - 1 ? "" : ", "); 
        }
        printf("\n");
    }


### 总结

这几年断断续续看了四五遍K&R的《TCPL》了，可一直都没写过C程序，现在开始多练习
练习吧。
