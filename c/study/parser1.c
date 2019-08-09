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


// 读取到指定位置
// fd 是描述字，buffer 是读缓冲区，bufsize 是缓冲区大小，flag 是结尾标志
// 返回值是读取到的字节数，offset 是 flag 的偏移量
static size_t readto(int fd, char *buffer, size_t bufsize, const char *flag, size_t *offset) {
    char *p = buffer; // 读缓冲区写入指针
    size_t total = 0; // 已读取字节数
    size_t m, n ;     // 每次欲读取字节数及实际读取字节数
    size_t lenflag;   // flag 长度
    int ret;          // indexof 结果

    lenflag = strlen(flag);
    m = BLOCKSIZE;
    *offset = -1;
    while ((n = read(fd, p, m)) > 0) {
        total += n;
        printf("debug: total=%d n=%d %s\n", total, n, hex(p, n));

        // 读取到标志
        ret = indexof(p - lenflag, lenflag + n, flag);
        if (ret != -1) {
            *offset = total - lenflag + ret;
            printf("read flag: total=%d ret=%d offset=%d\n", total, ret, *offset);
            break;
        }

        // buffer 满
        m = bufsize - total > BLOCKSIZE ? BLOCKSIZE : bufsize - total;
        if (m == 0) {
            fprintf(stderr, "buff full:%d %d\n", total, bufsize); 
            break;
        }

        p += n;
    }
    if (*offset == -1) *offset = total;
    return total;
}

int main() {
    char buffer[MAXSIZE]; // 缓冲区
    size_t offset;        // flag 的结尾偏移量
    size_t total = readto(STDIN_FILENO, buffer, MAXSIZE, "\r\n\r\n", &offset);
    printf("result total=%d, flag_offset=%d:\n%.*s\n", total, offset, offset, buffer);

    return 0;
}
