#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define listtype int 

struct list{
    int capacity;
    int size;
    listtype* data;
};

struct list* listinit(int capacity) {
    struct list *list = (struct list*)malloc(sizeof(struct list));
    list->capacity = capacity;
    list->size = 0;
    list->data = (listtype*)malloc(capacity * sizeof(listtype));
    return list;
}

int listfree(struct list *list) {
    free(list->data);
    free(list);
}

void listadd(struct list *list, listtype data) {
    listtype* newdata;
    if (list->size == list->capacity) {
        printf("auto expand:%d\n", list->size);
        list->capacity = list->capacity * 2;
        newdata = (listtype*)malloc(list->capacity * sizeof(listtype));
        // copy old to new
        memcpy(newdata, list->data, list->size * sizeof(listtype)); 
        free(list->data);
        list->data = newdata;
    }
    list->data[list->size] = data;
    list->size = list->size + 1;
}

void listprint(struct list *list) {
    printf("print list: %d/%d\n", list->size, list->capacity);
    int i;
    for (i = 0; i < list->size; i++) {
        printf("\t%d=%d\n", i, list->data[i]);
    }
}

int main(int argc, char* argv[]) {
    printf("hello 011\n");
    struct list *list = listinit(2);
    listadd(list, 'a');
    listadd(list, 'b');
    listadd(list, 'c');
    listadd(list, 'd');
    listadd(list, 'e');
    listprint(list);
    listfree(list);
    return 0;
}
