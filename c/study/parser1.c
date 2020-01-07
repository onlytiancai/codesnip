/*
 * 解析 http 应答
 * curl -sI baidu.com | ./parser1.o
 * */
#include <stdio.h>
#include <unistd.h>
#include <string.h>

#define MAXSIZE 1024
#define BLOCKSIZE 8 

// 查找子串的位置
static int indexof(const char *str, size_t lenstr, const char *substr) {
    int i, j;
    size_t lensubstr = strlen(substr);
    for (i = 0, j = 0; i < lenstr; i++) {
        j = str[i] == substr[j] ? j + 1 : 0;
        if (j == lensubstr) return i - j + 1;
    }

    return -1;
}

// 字节数组转换成 16 进制字符串
static char* hex(char *str, size_t n) {
    static char buff[128];
    char *p = buff;
    while (n--){
        p += snprintf(p, sizeof(buff) - (p - buff),   "%02X ", *str++); 
    }
    return buff;
}


// 读取到指定 flag 
// fd 是描述字，buffer 是读缓冲区，bufsize 是缓冲区大小，flag 是结尾标志
// offset 是 flag 末尾的偏移量, total 是读取到的总字节数
// 返回 0 表示读取到 flag，-1 表示缓冲区满
static int readto(int fd, char *buffer, size_t bufsize, const char *flag, 
        size_t *offset, size_t *total) {
    char *p = buffer; // 读缓冲区写入指针
    size_t m, n ;     // 每次欲读取字节数及实际读取字节数
    size_t lenflag;   // flag 长度
    int ret;          // indexof 结果

    lenflag = strnlen(flag, 8);
    m = BLOCKSIZE;
    *total = 0;
    *offset = 0;
    while(1) {
        n = read(fd, p, m);
        if (n <= 0) { // 流关闭
            fprintf(stderr, "read to eof:%d %d\n", *total, bufsize); 
            return -1;
        }

        *total += n;
        *offset += n;
        printf("debug: total=%d n=%d %s\n", *total, n, hex(p, n));

        // 回溯 lenflag 个字节，防止前面读取到半个 flag
        ret = indexof(p - lenflag, lenflag + n, flag);
        if (ret != -1) { // 读取到标志
            *offset = *total - n - lenflag + ret + lenflag;
            printf("read flag: total=%d ret=%d offset=%d\n", *total, ret, *offset);
            break;
        }

        m = bufsize - *total > BLOCKSIZE ? BLOCKSIZE : bufsize - *total;
        if (m == 0) { // buffer 满
            fprintf(stderr, "buff full:%d %d\n", *total, bufsize); 
            return -1;
        }

        p += n;
    }

    return 0;
}

int main() {
    char buffer[MAXSIZE]; // 缓冲区
    size_t offset;        // 读取到 flag 末尾的偏移量 
    size_t total;         // 读取的总字节数
    int ret = readto(STDIN_FILENO, buffer, MAXSIZE, "\r\n\r\n", &offset, &total);
    printf("result total=%d, flag_offset=%d:\n%.*s\n", total, offset, offset, buffer);

    return 0;
}
