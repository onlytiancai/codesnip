#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "list.h"

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

wlist *mklist(int item_size) {
    wlist *list= (wlist*)malloc(sizeof(wlist));
    list->size = 0;
    list->capacity = 3;
    list->item_size = item_size; 
    list->list =  (int*)malloc(list->capacity * list->item_size);
    list->push = &list_push;
    list->get = &list_get;
    return list;
}

void freelist(wlist *list) {
    free(list->list);
    free(list);
}
