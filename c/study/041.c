#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define N 10000

int cmpfunc (const void * a, const void * b)
{
   return ( *(int*)a - *(int*)b );
}

void random_fill(int* arr, int n)
{
    srand((unsigned)time(NULL));
    for (int i = 0; i < n; ++i) {
        arr[i] = rand() % N;
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

int array_eq(int* arr1, int* arr2, int n)
{
    for (int i = 0; i < n; ++i) {
       if (arr1[i] != arr2[i]) return 0;
    }
    return 1;
}

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

int main(int argc, char *argv[])
{
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

    printf("arr1 == arr2: %d\n", array_eq(arr1, arr2, N));
    printf("arr1 == arr3: %d\n", array_eq(arr1, arr3, N));

    return 0;
}
