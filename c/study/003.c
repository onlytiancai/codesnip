#include <stdio.h>

struct node {
    int data;
    struct node * next;
};

void printlinkedlist(const struct node* p) {
    while (p != NULL) {
        printf("%d %p\n", p->data, p->next);
        p = p->next;
    }
}

int main(int argc, int* argv[]) {
    printf("hello 003.\n");

    struct node n0;
    n0.data = 0;
    n0.next = NULL;
    printf("%d %p\n", n0.data, n0.next);

    struct node n1;
    n1.data = 1;
    n1.next = NULL;
    n0.next = &n1;
    printf("%d %p\n", n0.data, n0.next);
    printf("%d %p\n", n1.data, n1.next);

    struct node n2;
    n2.data = 2;
    n2.next = NULL;
    n1.next = &n2;
    printf("%d %p\n", n0.data, n0.next);
    printf("%d %p\n", n1.data, n1.next);
    printf("%d %p\n", n2.data, n2.next);

    printlinkedlist(&n0);
    

    // insert
    struct node n3;
    n3.data = 3;
    n3.next = &n2;
    n1.next = &n3;

    printlinkedlist(&n0);

    // remove
    n1.next = &n2;
    n3.next = NULL;
    printlinkedlist(&n0);

}
