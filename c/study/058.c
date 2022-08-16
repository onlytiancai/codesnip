#include <stdio.h>

int main()  
{  
    int a=1, b=2, c=0;  
  
    // 蛋疼的 add 操作  
    asm(  
        "addl %2, %0"       // 1  
        : "=g"(c)           // 2  
        : "0"(a), "g"(b)    // 3  
        : "memory");        // 4  
  
    printf("现在c是:%d\n", c);  
    return 0;  
}  
