#include <stdio.h>

void printarr(int* arr, int n) {
    int i;
    for (i = 0; i < n; i++) {
        printf("%d ", arr[i]); 
    }
    printf("\n");
}

int cmp(const void* a, const void* b) {
    int* x = (int*)a;
    int* y = (int*)b;
    return *x - *y;
}

int main(int argc, char* argv[]) {
    int arr[] = {8,9,1,5,7,4,3,6};
    int n = sizeof(arr) / sizeof(int);

    printarr(arr, n);
    qsort(arr, n, sizeof(int), cmp) ;
    printarr(arr, n);

    return 0;
}
