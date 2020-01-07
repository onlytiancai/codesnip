#ifndef WHASH_HEADER_FILE
#define WHASH_HEADER_FILE
typedef struct Whash{
    int capacity; // 哈希表容量
    int size; // 内容量
    char **keys; // 键数组
    void **values; // 值数组
    size_t *vlens; // 值的长度数组
    int (*put)(struct Whash *hash, char *k, void *v, size_t len); // 插入无类型数据
    int (*putstr)(struct Whash *hash, char *k, char *v); // 插入字符串
    int (*putint)(struct Whash *hash, char *k, int v); // 插入整数

    void* (*get)(struct Whash *hash, char *k); // 获取无类型数据
    char* (*getstr)(struct Whash *hash, char *k); // 获取字符串
    int (*getint)(struct Whash *hash, char *k); // 获取整数
} whash;

whash *mkhash();
void freehash(whash *hash);


#endif

