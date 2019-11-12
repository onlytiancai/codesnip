// gcc 004.c -lm && ./a.out
#include <stdio.h>
#include <math.h>

int main(int argc, char* argv[]) {
    printf("hello 004.\n");

    int i, j, n, b;
    n = 100;
    for (i = 2; i < n; i++) {
        b = 0;
        for (j = 2; j <= sqrt(i); j++) { // 一定是 <=
            if (i % j == 0) {
                b = 1;
                break;
            }
        }
        if (b == 0) {
            printf("%d 是质数\n", i); 
        }
    }

    return 0;
}
