#include <stdio.h>
#include <stdlib.h>

struct arr_list {
    int * arr;
    int index;
    int size;
};

struct arr_list* create_arr_list(n) {
    struct arr_list *arr = (struct arr_list*)malloc(sizeof(struct arr_list));
    arr->arr = (int*)malloc(sizeof(int) * n);
    arr->size = n;
    arr->index = 0;
    return arr;
}

int list_add(struct arr_list *arr, int obj) {
    int *tmp, i, *p, *q;
    // 空间不够用时，申请两倍空间，拷贝原数据到新空间里
    // 并把原空间释放掉
    if (arr->index >= arr->size) {
        tmp = (int *)malloc(sizeof(int) * arr->size * 2);
        p = arr->arr;
        q = tmp;
        for (i = 0; i < arr->index; i++) {
            *q++ = *p++;
        }
        free(arr->arr);
        arr->arr = tmp;
        arr->size = arr->size * 2;
    }
    arr->arr[arr->index++]= obj;
    return 0;
}

int list_get(const struct arr_list *arr, int index) {
    if (index < 0 || index >= arr->index) {
        return -1;
    }
    return arr->arr[index];
}

int free_arr_list(struct arr_list *arr){
    free(arr->arr);
    free(arr);
    return 0;
}

void print_arr_list(const struct arr_list *arr) {
    int i, t;
    printf("size=%d,index=%d\n", arr->size, arr->index);
    for (i = 0; i < arr->index; i++) {
        t = list_get(arr, i);
        printf("list[%d]=%d\n", i, t);
    }
}

int list_removeat(struct arr_list *arr, int index) {
    int i;
    if (index < 0 || index >= arr->index) {
        return -1;
    }
    for (i = index; i < arr->index - 1; i++) {
        arr->arr[index] = arr->arr[index + 1];
    }
    arr->index--;
    return 0;
}

int main(void)
{
    struct arr_list *arr;

    arr = create_arr_list(3);
    printf("list add: 5, 6, 7\n");
    list_add(arr, 5);
    list_add(arr, 6);
    list_add(arr, 7);
    print_arr_list(arr);


    printf("list add: 8, will auto expand\n");
    list_add(arr, 8);
    print_arr_list(arr);

    printf("list remove at 2\n");
    list_removeat(arr, 2);
    print_arr_list(arr);

    free_arr_list(arr);
    return 0;
}
