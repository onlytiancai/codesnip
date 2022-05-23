#include <stdio.h>
#define N 5

void mysort(int* arr, int n);
void print_arr(char* msg, int* arr, int n);

int main(int argc, char *argv[])
{
    int arr[] = {4, 1, 2, 3, 5};
    print_arr("排序前", arr, N);
    mysort(arr, N);
    print_arr("排序后", arr, N);

    return 0;
}

void print_arr(char* msg, int* arr, int n) 
{
    printf("%s\n", msg);
    for (int i = 0; i < n; ++i) {
       printf("%d\n", *arr++); 
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
