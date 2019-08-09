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

// 读取到指定位置, fd 是描述子，buffer 是读缓冲区，flag 是结尾标志
static size_t readto(int fd, char *buffer, const char *flag) {
    char *p = buffer; // 读缓冲区写入指针
    size_t total = 0; // 已读取字节数
    size_t m, n ;     // 每次欲读取字节数及实际读取字节数
    size_t lenflag;   // flag 长度

    lenflag = strlen(flag);
    m = BLOCKSIZE;
    while ((n = read(fd, p, m)) > 0) {
        total += n;
        printf("debug: total=%d n=%d\n", total, n);

        if (indexof(p - lenflag, lenflag + n, flag) != -1) {
            printf("read flag\n", total);
            break;
        }

        m = MAXSIZE - total > BLOCKSIZE ? BLOCKSIZE : MAXSIZE - total;
        if (m == 0) {
            fprintf(stderr, "buff full:%d %d\n", total, MAXSIZE); 
            break;
        }

        p += n;
    }

    return total;
}

int main() {
    char buffer[MAXSIZE];           // 读缓冲区
    int total = readto(STDIN_FILENO, buffer, "\r\n\r\n");
    printf("result:\n%.*s\n", total, buffer);

    return 0;
}
