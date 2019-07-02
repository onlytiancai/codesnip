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


// 返回负数表示找不到，0或正数表示找到
static int find_put_index(whash *hash, char* k) {
    unsigned int code = JSHash(k);
    int index, i;

    for (i = 0; i < hash->capacity; i++) {
        index = code % hash->capacity;
        char *p = hash->keys[index];
        // 找到空位或已存在相同的key
        if (p == NULL || strncmp(k, p, 100) == 0) {
            printf("find put index done:count=%d index=%d k=%s\n", i, index, k);
            return index;
        }
        code++;
    }

    printf("find put index failed:%s\n", k);
    return -1;
}

// 返回负数表示找不到，0或正数表示找到
static int find_get_index(whash *hash, char* k) {
    unsigned int code = JSHash(k);
    int index, i;

    for (i = 0; i < hash->capacity; i++) {
        index = code % hash->capacity;
        char *p = hash->keys[index];
        if (p != NULL && strncmp(k, p, 100) == 0) {
            printf("find get index done:count=%d index=%d k=%s\n", i, index, k);
            return index;
        }
        code++;
    }

    printf("find get index failed:%s\n", k);
    return -1;
}

// free 内部数据
static free_inner(int capacity, char **keys, char **values) {
    int i;
    for (i = 0; i < capacity; i++) {
        printf("free:%d %s %s\n", i, keys[i], values[i]);
        free(keys[i]);
        free(values[i]);
    }
    free(keys);
    free(values);
}

// 扩大 hashtable
static int extend(whash *hash) {
    // 保存原始数据
    char **keys = hash->keys;
    char **values = hash->values;
    int capacity = hash->capacity;

    // 重新分配内存
    hash->capacity = hash->capacity * 2;
    hash->keys = (char**)malloc(hash->capacity * sizeof(char*));
    memset(hash->keys, 0, hash->capacity * sizeof(char*));
    hash->values = (char**)malloc(hash->capacity * sizeof(char*));
    memset(hash->values, 0, hash->capacity * sizeof(char*));

    // rehash
    int i;
    printf("========= begin rehash\n");
    for (i = 0; i < capacity; i++) {
        hash->put(hash, keys[i], values[i]);
    }
    printf("========= end rehash\n");

    free_inner(capacity, keys, values);

}

static int hash_put(whash *hash, char *k, char *v) {
    printf("put: k=%s, v=%s\n", k, v);

    int index = find_put_index(hash, k);
    if (index < 0) {
        printf("hash full, will extend\n"); 
        extend(hash);
        index = find_put_index(hash, k);
    }

    // 保存 key
    size_t ksize = strnlen(k, 100);
    char* kcopy = (char*)malloc(ksize * sizeof(char)+1);
    kcopy[ksize * sizeof(char)] = '\0';
    strcpy(kcopy, k);
    hash->keys[index] = kcopy;

    // 保存 value
    size_t vsize = strnlen(v, 100);
    char* vcopy = (char*)malloc(vsize * sizeof(char)+1);
    vcopy[vsize * sizeof(char)] = '\0';
    strcpy(vcopy, v);
    hash->values[index] = vcopy;

    return 0;
}

static char* hash_get(whash *hash, char *k){
    int index = find_get_index(hash, k);
    if (index < 0) {
        return NULL; 
    }
    return hash->values[index];
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
    free_inner(hash->capacity, hash->keys, hash->values);
    free(hash);
}

int main() {

    char keys[4][10] = {"aaa", "bbb", "ccc", "ddd"};
    char values[4][10] = {"111", "222", "333", "444"};
    int size = 4;
    int i;
    whash *hash = mkhash();

    printf("========= begin put\n");
    for (i = 0; i < size; i++) {
        hash->put(hash, keys[i], values[i]);
    }

    printf("put duplicate key:%s %s", "aaa", "duplicate");
    hash->put(hash, "aaa", "duplicate");

    printf("========= begin get\n");
    for (i = 0; i < size; i++) {
        char *key = keys[i];
        char *value = hash->get(hash, key);
        printf("%s=%s\n", key, value);
    }
    printf("get notfound key:%s %s\n", "xxx", hash->get(hash, "xxx"));

    printf("========= begin free\n");
    freehash(hash);
    return 0;
}
