#include <stdio.h>
#include <time.h>

void printarr(int* arr, int n) {
    int i;
    for (i = 0; i < n; i++) {
        printf("%d ", arr[i]); 
    }
    printf("\n"); 
}

void sortarr(int* arr, int n) {
    int i, j, t;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n - i -1; j++ ){
            if (arr[j] > arr[j + 1]) {
                t = arr[j + 1];
                arr[j + 1] = arr[j];
                arr[j] = t;
            }
        }
    }
}

void randarr(int* arr, int n) {
    srand(time(NULL));
    int i;
    for (i = 0; i < n; i++) {
        arr[i] = rand() % n * 2;
    }
}

int main(int argc, char** argv) {
    int n = 20;
    int arr[n];

    randarr(arr, n);
    printarr(arr, n);

    sortarr(arr, n);
    printarr(arr, n);

    return 0;
}
