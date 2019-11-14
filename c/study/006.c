#include <stdio.h>

int main(int argc, char* argv[]) {
    printf("hello 006\n");
    int n = 5;
    int a[] = {1, 3, 5, 7, 9};
    int b[] = {2, 4, 6, 8, 10};
    int c[n];
    int i;

    for (i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }

    for (i = 0; i < n; i++) {
        printf("%d %d\n", i, c[i]);
    }

    int *pa = a, *pb = b;
    for (i = 0; i < n; i++) {
        c[i] = *pa++ + *pb++;
    }

    for (i = 0; i < n; i++) {
        printf("%d %d\n", i, c[i]);
    }

    return 0;
}
