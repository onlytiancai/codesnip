#include <stdio.h>

#define N 11

int main() {
    printf("hello 013\n");
    int i, j;
    for (i = 0; i < N; i+=2) {
        for (j = 0; j < (N - i) / 2; j++) {
            printf(" ");
        }

        for (j = 0; j <= i; j++) {
            printf("*");
        }

        printf("\n");
    }
    for (i = N - 2; i > 0; i-=2) {
        for (j=(N - i) / 2; j > 0; j--){
            printf(" ");
        }
        for (j = i; j > 0; j-- ){
            printf("*");
        }
        printf("\n");
    } 

    return 0;
}
