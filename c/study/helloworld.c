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

// for循环练习
static void n_hello_world(int n){
    int i = 0;
    for (i = 0; i < n; i++) {
        printf("hello, world\n");
    }
}

// 字符串练习
static int w_strlen(const char* str){
    int i;
    for (i = 0; *str != '\0'; str++, i++) {
        ;
    }
    return i;
}

// 指针练习
static char* reversal(const char* str){
    const char* p;
    //TODO: 谁来释放呢？
    char* ret = (char*)malloc(strlen(str) * sizeof(char));

    p = str + strlen(str);
    
    while (p-- == str){
        printf("%c\n", *p);
    }

    return ret;
}

int main(){
    //hello_world();
    //n_hello_world(5);
    //printf("len=%d\n", w_strlen("hello world"));
    printf("%s", reversal("hello"));
    return 0;
}
