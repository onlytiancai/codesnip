#include <stdio.h>
#include <unistd.h>
#include <string.h>

#define MAXSIZE 1024
#define BLOCKSIZE 8 

static int indexof(const char *str, const size_t lenstr, 
        const char *substr, const size_t lensubstr) {
    int i, j;

    for (i = 0, j = 0; i < lenstr; i++) {
        j = str[i] == substr[j] ? j + 1 : 0;
        if (j == lensubstr) return i - j + 1;
    }

    return -1;
}

int main() {
    char header[MAXSIZE];
    char *p = header;
    size_t total = 0;
    size_t m, n ;
    char flag[] = "\r\n\r\n"; 
    size_t lenflag = strlen(flag);

    m = BLOCKSIZE;
    while ((n = read(STDIN_FILENO, p, m)) > 0) {
        total += n;
        printf("%d %.*s\n", total, n, p);

        if (indexof(p-lenflag, lenflag + n, flag, lenflag) != -1) {
            printf("read flag\n%.*s", total, header); 
        }

        m = MAXSIZE - total > BLOCKSIZE ? BLOCKSIZE : MAXSIZE - total;
        if (m == 0) {
            fprintf(stderr, "buff full:%d %d\n", total, MAXSIZE); 
            break;
        }

        p += n;
    }
    return 0;
}
