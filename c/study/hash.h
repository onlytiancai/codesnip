#ifndef WHASH_HEADER_FILE
#define WHASH_HEADER_FILE
typedef struct Whash{
    int capacity;
    int size;
    int *list;
    int (*push)(struct Whash *hash, int n);
    int (*get)(struct Whash *hash, int n);
} whash;

whash *mkhash();
void freehash(whash *hash);

#endif

