#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>

int foo(int i) {
    int n = rand() % 500000 + 300000;
    usleep(n);
    printf("%d\n", i);
    return n;
}

int main(int argc, char *argv[])
{
    srand(time(0));
    for (int i = 0; i < 10000; ++i) {foo(i);}    
    return 0;
}
