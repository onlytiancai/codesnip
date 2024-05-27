#include <stdio.h>

struct data
{
    char a;
    int b;
    char c;
};
int main(int argc, char *argv[])
{
    struct data d = {1,2,3};
    printf("%p %p %p\n", &d.a, &d.b, &d.c);
    return 0;
}
