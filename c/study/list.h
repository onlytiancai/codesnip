#ifndef WLIST_HEADER_FILE
#define WLIST_HEADER_FILE

typedef struct Wlist {
    int capacity;
    int size;
    int item_size;
    int *list;
    int (*push)(struct Wlist *list, int n);
    int (*get)(struct Wlist *list, int n);
} wlist;

wlist *mklist();
void freelist(wlist *list);

#endif

