#include <stdio.h>
#include "hash.h"

// gcc -c hash.c
// gcc -o testhash.o test_hash.c hash.o
// ./testhash.o

int main() {
    char keys[4][10] = {"aaa", "bbb", "ccc", "ddd"};
    char values[4][10] = {"111", "222", "333", "444"};
    int size = 3;
    int i;

    whash *hash = mkhash();
    for (i = 0; i < size; i++) {
        hash->putstr(hash, keys[i], values[i]);
    }

    for (i = 0; i < size; i++) {
        char *key = keys[i];
        char *value = hash->getstr(hash, key);
        printf("%s=%s", key, value);
    }

    freehash(hash);
    return 0;
}
