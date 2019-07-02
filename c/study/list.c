#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int wlist;

struct ListNode {
    int capacity;
    int size;
    int item_size;
    int *list;
};

static struct ListNode *list_table[10];

wlist mklist() {
    static int list_seq = 0;

    struct ListNode *node = (struct ListNode*)malloc(sizeof(struct ListNode));
    node->size = 0;
    node->capacity = 3;
    node->item_size = sizeof(int);
    node->list =  (int*)malloc(node->capacity * node->item_size);
    list_table[list_seq] = node;

    return list_seq++;
}

void list_free(wlist list) {
    struct ListNode *pnode = list_table[list];
    free(pnode->list);
    free(pnode);
}

int list_push(wlist list, int n) {
    // TODO: table size check
    struct ListNode *node = list_table[list];
    printf("push :%d %d\n", node->size, node->capacity);
    if (node->size >= node->capacity) {
        printf("capacity extend :%d %d\n", node->size + 1, node->capacity);

        node->capacity = node->capacity * 2;
        int *origin_list = node->list;
        node->list =  (int*)malloc(node->capacity * node->item_size);
        memcpy(node->list, origin_list, node->size * node->item_size);
        free(origin_list);
    }
    node->list[node->size++] = n;
}

int list_get(wlist list, int n) {
    struct ListNode *node = list_table[list];
    if (n >= node->capacity) {
        printf("get error\n");
        exit(1);
    }
    return node->list[n];
}

int list_size(wlist list) {
    struct ListNode *node = list_table[list];
    return node->size;
}

int main() {
    wlist list = mklist();
    printf("mklist: %d\n", list);

    int i, size;
    for (i = 0; i < 20; i++) {
        list_push(list, i);
    }

    size = list_size(list);
    for (i = 0; i < size; i++) {
        printf("list[%d] = %d\n", i, list_get(list, i)); 
    } 
    list_free(list);
    return 0;
}
