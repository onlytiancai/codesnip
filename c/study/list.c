#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "list.h"

static int list_push(wlist *list, int n) {
    int *origin_list;
    printf("push :%d %d\n", list->size, list->capacity);


    if (list->size >= list->capacity) {
        origin_list = list->list;
        printf("capacity extend :%d %d\n", list->size + 1, list->capacity);

        list->capacity = list->capacity * 2;
        list->list =  (int*)malloc(list->capacity * list->item_size);
        if (list->list != NULL) {
            memcpy(list->list, origin_list, list->size * list->item_size);
        } else {
            printf("malloc error");
            free(origin_list);
            return -1;
        }
        free(origin_list);
    }
    if (list->list != NULL){
        list->list[list->size++] = n;
    } else {
        printf("malloc error");
        return -1;
    }
    return 0;
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
