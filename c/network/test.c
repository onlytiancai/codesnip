#include <stdlib.h>
#include <stdio.h>
#include <string.h>


int main(int argc, const char *argv[])
{
    char* a = "123";
    printf("a len = %d \n", strlen(a)); // not including the terminating '\0' character.

    char buff[3];
    buff[0] = '1';
    buff[1] = '2';
    buff[2] = '3';
    buff[3] = '\0'; // 真是毁了三观了，这么明显的问题编译器检查不出来呀。
    printf("buff len = %d \n", strlen(buff));

    int i = 0, n = 3;
    char buff2[n];
    for(i = 0; i < n; i ++){
        printf("default %d = %d \n", i, buff2[i]); 
    }

    memset(buff2, 0, n);
    for(i = 0; i < n; i ++){
        printf("after memset %d = %d \n", i, buff2[i]); 
    }

    return 0;
}
