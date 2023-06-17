#include <stdio.h>
#include <stdlib.h>

struct Node {
    int a;
    int b;
};
struct Node *nodes = NULL;
int main(int argc, char *argv[])
{
    nodes = (struct Node*)malloc(10000*sizeof(struct Node));
    printf("000: %p %p %p\n", nodes, &nodes[0], &(nodes[0].a));
    struct Node *n;
    for (int i = 0; i < 2; ++i) {
        n = &nodes[i];
        n->a = i;
        n->b = i;
    }
    int *pi = &(nodes[0].a);
    *pi = 5;
    nodes = realloc(nodes, 20000*sizeof(struct Node));
    printf("111: %p %p %p\n", nodes, &nodes[0], &(nodes[0].a));

    free(nodes);
    return 0;
}
