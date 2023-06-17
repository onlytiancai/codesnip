#include <stdio.h>
#include <stdlib.h>

struct sort_item {
    float data;
    int index;
};

int cmpfunc (const void *a, const void *b)
{
    float x =  ((struct sort_item*)a)->data;
    float y =  ((struct sort_item*)b)->data;
    printf("debug: %f-%f=%f\n", x,y,x-y);
    if (x>y) return 1;
    if (x<y) return -1;
    return 0;
}

int *argsort(int n, float *arr, int *ret)
{
    struct sort_item *items = (struct sort_item*)malloc(n*sizeof(struct sort_item));
    if (items == NULL) {
        perror("realloc, error");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < n; ++i) {
        items[i].data = arr[i];
        items[i].index= i;
    }
    qsort(items, n, sizeof(struct sort_item), cmpfunc);
    for (int i = 0; i < n; ++i) {
        ret[i] = items[i].index;
        printf("debug2 %f %d\n", items[i].data, i);
    }
    free(items);
}

int main(int argc, char *argv[])
{
    float arr[] = {3.87,5,4.58,0,0,4.12};
    int n = sizeof(arr)/sizeof(float);
    int ret[n];
    argsort(n, arr, ret);

    for (int i = 0; i < n; ++i) {
        printf("%d ", ret[i]);
    }
    printf("\n");

    return 0;
}
