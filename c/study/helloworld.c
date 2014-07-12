// 检查:
//      splint helloworld.c
// 编译:
//     gcc helloworld.c -o helloworld.o 
// 执行：
//       ./helloworld.o
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 函数练习
static void hello_world(){
    printf("hello, world\n");
}

// for循环练习, 打印n次hello world
static void n_hello_world(int n){
    int i = 0;
    for (i = 0; i < n; i++) {
        printf("hello, world\n");
    }
}

// 字符串练习, 计算字符串长度
static int w_strlen(const char* str){
    int i;
    for (i = 0; *str != '\0'; str++, i++) {
        ;
    }
    return i;
}

static void change_str_test(){
    //char* s1 = "hello"; // will core dump
    char s1[10] = "hello";
    *s1 = 'p';
    printf("%s\n", s1);
}

// 指针练习, 反转字符串
static char* reversal(char* str){
    char* ret = str;
    char* p = str + strlen(str) - 1;
    char c;
   
    for ( ; p > str; --p, ++str) { 
        printf("debug:reversal: %p %p %c %c\n", p, str, *p, *str);
        c = *p;
        *p = *str;
        *str = c;
    }

    return ret;
}

int main(){
    char str[10] = "hello";
    hello_world();

    n_hello_world(5);

    printf("len=%d\n", w_strlen("hello world"));

    change_str_test();

    printf("222 %s\n", reversal(str));

    return 0;
}
