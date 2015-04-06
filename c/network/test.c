#include <stdlib.h>
#include <stdio.h>
#include <string.h>


int main(int argc, const char *argv[])
{
    char* a = "123";
    printf("a len = %d \n", strlen(a)); // not including the terminating '\0' character.

    char buff[4];
    buff[0] = '1';
    buff[1] = '2';
    buff[2] = '3';
    buff[3] = '\0';
    printf("buff len = %d \n", strlen(buff));

    return 0;
}
