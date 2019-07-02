#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// #include "hash.h"

typedef struct Whash{
    int capacity;
    int size;
    char **keys;
    char **values;
    int (*put)(struct Whash *hash, char *k, char *v);
    char* (*get)(struct Whash *hash, char *k);
} whash;

whash *mkhash();
void freehash(whash *hash);


// JS Hash Function  
static unsigned int JSHash(char *str)  
{  
    unsigned int hash = 1315423911;  
   
    while (*str)  
    {  
        hash ^= ((hash << 5) + (*str++) + (hash >> 2));  
    }  
   
    return (hash & 0x7FFFFFFF);  
}

static int hash_put(whash *hash, char *k, char *v) {
    unsigned int code = JSHash(k);
    int kindex = code % hash->capacity;
    printf("put:code=%d, index=%d, k=%s, v=%s\n", code, kindex, k, v);

    // 解决冲突 
    char *kfound = hash->keys[kindex];
    if (kfound != NULL) {
        printf("hash conflict:%d %s %s\n", code, k, kfound); 
    }

    // 保存 key
    size_t ksize = strnlen(k, 100);
    char* kcopy = (char*)malloc(ksize * sizeof(char)+1);
    kcopy[ksize * sizeof(char)] = '\0';
    strncpy(kcopy, k, 100);
    hash->keys[kindex] = kcopy;

    // 保存 value
    size_t vsize = strnlen(v, 100);
    char* vcopy = (char*)malloc(vsize * sizeof(char)+1);
    vcopy[vsize * sizeof(char)] = '\0';
    strncpy(vcopy, v, 100);
    hash->values[kindex] = vcopy;

    return 0;
}

static char* hash_get(whash *hash, char *k){
    unsigned int code = JSHash(k);
    int kindex = code % hash->capacity;
    printf("get:code=%d, index=%d\n", code, kindex);

    // TODO: not found
    return hash->values[kindex];
}

whash *mkhash() {
    whash *hash = (whash*)malloc(sizeof(whash));
    hash->size = 0;
    hash->capacity = 3;

    // 申请keys，values 内存，并清空
    hash->keys = (char**)malloc(hash->capacity * sizeof(char*));
    memset(hash->keys, 0, hash->capacity * sizeof(char*));

    hash->values = (char**)malloc(hash->capacity * sizeof(char*));
    memset(hash->values, 0, hash->capacity * sizeof(char*));

    hash->put = &hash_put; 
    hash->get = &hash_get; 
    return hash;
}

void freehash(whash *hash) {

}

int main() {

    char keys[4][10] = {"aaa", "bbb", "ccc", "ddd"};
    char values[4][10] = {"111", "222", "333", "444"};
    int size = 4;
    int i;

    whash *hash = mkhash();
    for (i = 0; i < size; i++) {
        hash->put(hash, keys[i], values[i]);
    }

    for (i = 0; i < size; i++) {
        char *key = keys[i];
        char *value = hash->get(hash, key);
        printf("%s=%s\n", key, value);
    }

    freehash(hash);
    return 0;
}
