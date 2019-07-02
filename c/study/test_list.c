#include <stdio.h>
#include "list.h"

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
