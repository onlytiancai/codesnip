#ifndef WLIST_HEADER_FILE
#define WLIST_HEADER_FILE

typedef struct Wlist {
    int capacity;
    int size;
    size_t item_size;
    int *list;
    int (*push)(struct Wlist *list, int n);
    int (*get)(struct Wlist *list, int n);
} wlist;

wlist *mklist(int item_size);
void freelist(wlist *list);

#endif

