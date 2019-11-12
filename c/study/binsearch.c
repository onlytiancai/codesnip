#include <stdio.h>
#include <time.h>

#define N 20

int randint(int n) {
    return rand() % n;
}

void printarr(int arr[], int n) {
    int i;
    for (i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

int* randarr() {
    static int arr[N];
    int i = 0;
    for (i = 0; i < N; i++) {
        arr[i] = randint(N*2);
    }
    return arr; 
}

int* sortarr(int* arr, int n) {
    int i, j, t;
    for (i = 0; i < n; i++) {
        for(j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                t = arr[j + 1];
                arr[j + 1] = arr[j];
                arr[j] = t;
            }
        }
        // printf("debug: n-i-1=%i arr last=%d\n", n-i-1, arr[n-i-1]);
    }
    return arr;
}

int main(int argc, char* argv[]) {
    srand(time(NULL));

    int* arr = randarr();
    printarr(arr, N);

    arr = sortarr(arr, N);
    printarr(arr, N);

    int x = argc == 2 ? atoi(argv[1]) : randint(N*2);
    printf("n=%d x=%d\n", N, x);

    int i, m, l, r;
    i = 0;
    l = 0; 
    r = N -1;
    while (l < r) {
        m = (r - l) / 2 + l;
        printf("debug:l=%d(%d), m=%d(%d), r=%d(%d) x=%d\n", l, arr[l],  m, arr[m], r, arr[r], x);
        if (x < arr[m]) {
            r = m - 1;
        } else if (x > arr[m]) {
            l = m + 1;
        } else {
            printf("x found\n");
            return 0;
        }
        if (++i > N) {
            printf("overflow\n");
            break; 
        }
    }
    printf("x not found\n");

    return 0;
}
