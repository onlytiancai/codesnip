#include <stdio.h>
#include <unistd.h>

#define MAXSIZE 9 
#define BLOCKSIZE 4

int main() {
    char header[MAXSIZE];
    char *p = header;
    size_t total = 0;
    size_t m, n ;
    char flag[] = "\r\n\r\n"; 

    m = BLOCKSIZE;
    while ((n = read(STDIN_FILENO, p, m)) > 0) {
        total += n;
        printf("%d %.*s\n", total, n, p);

        m = MAXSIZE - total > BLOCKSIZE ? BLOCKSIZE : MAXSIZE - total;
        if (m <= 0) {
            fprintf(stderr, "read overflow:%d\n", total); 
            return -1;
        }

        p += n;
    }
    return 0;
}
