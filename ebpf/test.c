#include <stdio.h>
//struct AAA { int xxx; char *yyy; int zzz; struct BBB *ooo; };
//struct BBB {int xxx; int yyy;};
struct AAA {
    int xxx;
    char *yyy;
    int zzz;
    struct BBB *ooo;
};
struct BBB {
    int xxx;
    int yyy;
};



int main(int argc, char *argv[])
{
    struct AAA aaa;
    aaa.yyy = "hello";
    printf("%s\n", aaa.yyy);
    return 0;
}
