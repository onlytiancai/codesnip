#include <stdio.h>

int main(int argc, char *argv[])
{
    int arr[] = {31, -41, 59, 26, -53, 58, 97, -93, -23, 84};
    int i = 0, j = 0, k = 0, sum = 0, maxsofar = 0;
    int n = sizeof(arr)/sizeof(int);

    for (int i = 0; i < n; ++i) {
        for (j = i; j < n; ++j) {
            sum = 0;
            for (k = i; k <= j; k++) {
               sum += arr[k];
               if (sum > maxsofar) {
                   maxsofar = sum;
               }
            }
        }
    }
    printf("maxsofar=%d\n", maxsofar);
    return 0;
}
