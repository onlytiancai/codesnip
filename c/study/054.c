#include <stdio.h>
#include <stdint.h>

struct Node {
    int a;
    int b;
};
int main(int argc, char *argv[])
{
    struct Node nodes[10];
    nodes[5].a = 1;
    nodes[5].b = 2;

    struct Node *p = &nodes[0];

    printf("%d %d\n", (p+5)->a, (p+5)->b);
    (p+5)->a = 6;
    printf("%d %d\n", (p+5)->a, (p+5)->b);
    int *ip = &((p+5)->a);
    *ip = 7;
    printf("%d %d\n", (p+5)->a, (p+5)->b);

    uintptr_t diff = (uintptr_t)ip-(uintptr_t)p;
    printf("111 %ld\n", diff);
    int * new_ip = (int*)((uintptr_t)p+diff); 
    printf("222 %p %p\n", ip, new_ip);
    *new_ip = 8;
    printf("%d %d\n", (p+5)->a, (p+5)->b);


    return 0;
}
