#include <stdio.h>

#define LEN 10

static int arr[] = {1,3,5,7,9,11,13,15,17,18};

int print_r(int *arr, int len)
{
    int i;
    for (i = 0; i < len; i++ ) {
        printf("%d %d\n", i, arr[i]);
    }
}

int search(int *arr, int len, int t)
{
    int l = 0, u = len - 1, m;

    while (l <= u) {
        m = (l + u) / 2;
        printf("debug: %d %d %d \n", l, m , u);
        if (t < arr[m]) {
            u = m - 1; 
        } else if (t == arr[m]) {
            return m;
        } else {
            l = m + 1;
        }
    }

    return -1;
}

int main(int argc, char *argv[])
{

    if (argc != 2) {
        printf("Usage: %s t\n", argv[0]); 
        return -1; 
    }

    int ret, t = atoi(argv[1]);

    print_r(arr, LEN);

    ret = search(arr, LEN, t);
    if (ret != -1) printf("find ok: %d %d \n", t, ret);
    else printf("not find: %d \n", t);

    return 0;
}
