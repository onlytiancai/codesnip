#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 10

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

int main(int argc, char* argv[]) {
    printf("hello 005\n");
    
    srand(time(NULL));
    struct node *head, *p, *prev;
    prev = NULL;
    int i;
    for (i = 0; i < N; i++) {
        p = (struct node*)malloc(sizeof(struct node));
        p->data = rand() % N * 2;
        p->next = NULL;
        printf("debug: %d %p\n", p->data, p);
        if (i == 0) head = p;
        if (prev != NULL) prev->next = p;
        prev = p;
    }
    printlinkedlist(head);

    return 0;
}
