#include <stdio.h>
#include "hash.h"

int main() {

    char *keys[] = ["aaa", "bbb", "ccc"];
    char *values[] = ["111", "222", "333"];
    int size = 3;
    int i;

    whash *hash = mkhash();
    for (i = 0; i < size; i++) {
        hash->put(keys[i], values[i]);
    }

    for (i = 0; i < size; i++) {
        char *key = keys[i];
        char *value = hash->get(key);
        printf("%s=%s", key, value);
    }

    freehash(hash);
    return 0;
}
