#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#define N 10000




void array_copy(int* arr1, int* arr2, int n)
{
    for (int i = 0; i < n; ++i) {
       arr2[i] = arr1[i];
    }
}

void array_print(char* msg, int* arr, int n)
{
    printf("%s\n", msg);
    for (int i = 0; i < n; ++i) {
        printf("%d\n", arr[i]); 
    }
}

void mysort(int* arr, int n)
{
    int i, j, t;
    for (i = 0; i < n; ++i) {
        for (j = i;  j > 0 && arr[j-1] > arr[j]; j--) {
            t = arr[j-1]; arr[j-1] = arr[j]; arr[j] = t;
        } 
    } 
}

void mysort2(int* arr, int l, int u)
{
    if (l >= u) return;
    int m = l, i, t;
    for (i = l+1; i <= u; ++i) {
        if (arr[i] < arr[l]) {
            t = arr[++m]; arr[m] = arr[i]; arr[i] = t;
        }
    }

    t = arr[l]; arr[l] = arr[m]; arr[m] = t;

    mysort2(arr, l, m-1);
    mysort2(arr, m+1, u);
}

int cmpfunc (const void * a, const void * b) {
   return ( *(int*)a - *(int*)b );
}

void random_fill(int* arr, int n) {
    srand((unsigned)time(NULL));
    for (int i = 0; i < n; ++i) { arr[i] = rand() % (N * 2); }
}

int array_equal(int* arr1, int* arr2, int n) {
    for (int i = 0; i < n; ++i) {
       if (arr1[i] != arr2[i]) return 0;
    }
    return 1;
}

int main(int argc, char *argv[]) {
    clock_t start, end;
    int arr1[N], arr2[N], arr3[N];
    random_fill(arr1, N);
    array_copy(arr1, arr2, N);
    array_copy(arr1, arr3, N);

    start = clock();
    qsort(arr1, N, sizeof(int), cmpfunc);
    end = clock();
    printf("qsort time cost=%fms\n",(double)(end-start)/CLOCKS_PER_SEC*1000);

    start = clock();
    mysort(arr2, N);
    end = clock();
    printf("mysort time cost=%fms\n",(double)(end-start)/CLOCKS_PER_SEC*1000);

    start = clock();
    mysort2(arr3, 0, N-1);
    end = clock();
    printf("mysort2 time cost=%fms\n",(double)(end-start)/CLOCKS_PER_SEC*1000);

    assert(array_equal(arr1, arr2, N));
    assert(array_equal(arr1, arr3, N));

    return 0;
}
