#include <stdio.h>

// 对参数 a 和 b 求和
int sum(int a, int b)
{
        int s=a+b;
        return s;
}

// main函数：程序入口
int main(int argc, char* argv[])
{
        int n=sum(1, 2); // 调用sum函数对求和
        printf("n: %d\n", n);  //在屏幕输出 n 的值
        return 0;
}
