#include <stdio.h>

int main(int argc, char* argv[]) {
    printf("hello 010\n");
    int i, j;
    int map[91];
    for (i = 0; i < 91; i++) {
        map[i] = 0;
    }
    for (i = 1; i < 10; i++) {
        for (j = i; j < 10; j++) {
            if (map[i * j] == 0) {
                printf("%d*%d=%d ", i, j, i * j);
                map[i * j] = 1;
            }
        }
        printf("\n");
    }
    return 0;
}
