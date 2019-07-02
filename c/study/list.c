#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct Wlist {
    int capacity;
    int size;
    int item_size;
    int *list;
    int (*push)(struct Wlist *list, int n);
    int (*get)(struct Wlist *list, int n);
} wlist;

static int list_push(wlist *list, int n) {
    printf("push :%d %d\n", list->size, list->capacity);
    if (list->size >= list->capacity) {
        printf("capacity extend :%d %d\n", list->size + 1, list->capacity);

        list->capacity = list->capacity * 2;
        int *origin_list = list->list;
        list->list =  (int*)malloc(list->capacity * list->item_size);
        memcpy(list->list, origin_list, list->size * list->item_size);
        free(origin_list);
    }
    list->list[list->size++] = n;
}

static int list_get(wlist *list, int n) {
    if (n < 0 || n >= list->capacity) {
        printf("get error:size=%d n=%d\n", list->size, n);
        exit(1);
    }
    return list->list[n];
}

wlist *mklist() {
    wlist *list= (wlist*)malloc(sizeof(wlist));
    list->size = 0;
    list->capacity = 3;
    list->item_size = sizeof(int);
    list->list =  (int*)malloc(list->capacity * list->item_size);
    list->push = &list_push;
    list->get = &list_get;
    return list;
}

void freelist(wlist *list) {
    free(list->list);
    free(list);
}
int main() {
    wlist *list = mklist();
    printf("mklist: %p\n", list);

    int i ;
    for (i = 0; i < 10; i++) {
        list->push(list, i);
    }

    for (i = 0; i < list->size; i++) {
        printf("list[%d] = %d\n", i, list->get(list, i)); 
    } 
    freelist(list);
    return 0;
}
