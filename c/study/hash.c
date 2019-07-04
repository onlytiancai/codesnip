#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct Whash{
    int capacity; // 哈希表容量
    int size; // 内容量
    char **keys; // 键数组
    void **values; // 值数组
    size_t *vlens; // 值的长度数组
    int (*putstr)(struct Whash *hash, char *k, char *v); // 插入字符串
    int (*putint)(struct Whash *hash, char *k, int v); // 插入整数
    char* (*getstr)(struct Whash *hash, char *k); // 获取字符串
    int (*getint)(struct Whash *hash, char *k); // 获取整数
} whash;

whash *mkhash();
void freehash(whash *hash);

static void* hash_get(whash *hash, char *k);
static int hash_put_voidp(whash *hash, char *k, void *v, size_t len);

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


// 查找可以插入 key 的空位置
// 返回负数表示找不到，非负数表示找到
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

// 查找已存在的 key 的位置，会比较 hashcode，以及进行字符串对比
// 返回负数表示找不到，非负表示找到
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

// 释放内部数据
static free_inner(int capacity, char **keys, void **values, size_t *vlens) {
    int i;
    for (i = 0; i < capacity; i++) {
        printf("free:%d %s\n", i, keys[i]);
        free(keys[i]);
        free(values[i]);
    }
    free(keys);
    free(values);
    free(vlens);
}

// 扩大 hashtable
static int extend(whash *hash) {
    // 保存原始数据
    char **keys = hash->keys;
    void **values = hash->values;
    size_t *vlens = hash->vlens;
    int capacity = hash->capacity;

    // 重新分配内存
    hash->capacity = hash->capacity * 2;
    hash->keys = (char**)malloc(hash->capacity * sizeof(char*));
    memset(hash->keys, 0, hash->capacity * sizeof(char*));
    hash->values = (void**)malloc(hash->capacity * sizeof(void*));
    memset(hash->values, 0, hash->capacity * sizeof(void*));
    hash->vlens = (size_t*)malloc(hash->capacity * sizeof(size_t));
    memset(hash->vlens, 0, hash->capacity * sizeof(size_t));

    // rehash
    int i;
    printf("========= begin rehash\n");
    for (i = 0; i < capacity; i++) {
        hash_put_voidp(hash, keys[i], values[i], vlens[i]);
    }
    free_inner(capacity, keys, values, vlens);
    printf("========= end rehash\n");

}

static int hash_put_voidp(whash *hash, char *k, void *v, size_t len) {
    printf("put: k=%s vlen=%d\n", k, len);

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
    void *vcopy = malloc(len);
    memcpy(vcopy, v, len);
    hash->values[index] = vcopy;

    // 保存值长度
    hash->vlens[index] = len;

    return 0;
}


static int hash_putstr(whash *hash, char *k, char *v) {
    printf("putstr: k=%s, v=%s\n", k, v);
    size_t vsize = strnlen(v, 100);
    return hash_put_voidp(hash, k, v, vsize + 1);
}


static void* hash_get(whash *hash, char *k){
    int index = find_get_index(hash, k);
    if (index < 0) {
        return NULL; 
    }
    return (void*)hash->values[index];
}

static char* hash_getstr(whash *hash, char *k){
    return (char*)hash_get(hash, k);
}

static int hash_getint(whash *hash, char *k){
    int *ret = (int*)hash_get(hash, k);
    return *ret;
}

whash *mkhash() {
    whash *hash = (whash*)malloc(sizeof(whash));
    hash->size = 0;
    hash->capacity = 3;

    // 申请keys，values, vlens 内存，并清空
    hash->keys = (char**)malloc(hash->capacity * sizeof(char*));
    memset(hash->keys, 0, hash->capacity * sizeof(char*));

    hash->values = malloc(hash->capacity * sizeof(void*));
    memset(hash->values, 0, hash->capacity * sizeof(void*));

    hash->vlens= malloc(hash->capacity * sizeof(size_t));
    memset(hash->vlens, 0, hash->capacity * sizeof(size_t));

    hash->putstr = &hash_putstr; 
    hash->getstr = &hash_getstr; 
    return hash;
}

void freehash(whash *hash) {
    free_inner(hash->capacity, hash->keys, hash->values, hash->vlens);
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
        hash->putstr(hash, keys[i], values[i]);
    }

    printf("put duplicate key:%s %s", "aaa", "duplicate");
    hash->putstr(hash, "aaa", "duplicate");

    printf("========= begin get\n");
    for (i = 0; i < size; i++) {
        char *key = keys[i];
        char *value = hash->getstr(hash, key);
        printf("%s=%s\n", key, value);
    }
    printf("get notfound key:%s %s\n", "xxx", hash->getstr(hash, "xxx"));

    printf("========= begin free\n");
    freehash(hash);
    return 0;
}
