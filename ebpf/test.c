#include <stdio.h>
struct BBB {int xxx; int yyy;};
struct AAA { int xxx; char yyy; int *zzz; struct BBB *ooo; };


int main(int argc, char *argv[])
{
    struct AAA aaa;
    return 0;
}
