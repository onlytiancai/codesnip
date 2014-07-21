#include <stdio.h>
#include <stdlib.h>

struct arr_list {
    int * arr;      // 内部数组
    int index;      // 实际数据大小
    int size;       // 预分配空间大小
};

// 创建一个array list
struct arr_list* create_arr_list(n) {
    if (n < 1) {
        n = 10;
    }
    struct arr_list *arr = (struct arr_list*)malloc(sizeof(struct arr_list));
    arr->arr = (int*)malloc(sizeof(int) * n);
    arr->size = n;
    arr->index = 0;
    return arr;
}

// 空间不足时自动扩容，默认策略是空间不够时申请双倍大小空间
// 然后把原有数据拷贝到新空间，并把原有空间释放掉
static void expand_space(struct arr_list *arr) {
    int *tmp, i, *p, *q;

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
}

// 在指定位置插入新元素，现有元素向右移，O(N)
int list_insert(struct arr_list *arr, int index, int obj) {
    int i;

    if (index < 0 || index > arr->index) {
        return -1;
    }
    expand_space(arr);

    for (i = arr->index; i > index ; i--) {
        arr->arr[i] = arr->arr[i - 1];
    }
    arr->arr[index] = obj;
    arr->index++;
    return 0;
}

// 在array list 末尾插入数据， O(1)
int list_push(struct arr_list *arr, int obj) {
    return list_insert(arr, arr->index, obj);
}

// 获取array list指定位置的数据, O(1),
// 成功返回0，失败返回非0, 获取得到的数据保存在obj指针里
int list_get(const struct arr_list *arr, int index, int *obj) {
    if (index < 0 || index >= arr->index) {
        return -1;
    }
    *obj = arr->arr[index];
    return 0;
}

// 设置array list指定位置的数据, O(1)
// 成功返回0，失败返回非0
int list_set(struct arr_list *arr, int index, int obj) {
    if (index < 0 || index >= arr->index) {
        return -1;
    }
    arr->arr[index] = obj;
    return 0;
}

// 释放一个array list的内存
int free_arr_list(struct arr_list *arr){
    free(arr->arr);
    free(arr);
    return 0;
}

// 从头打印一个arr list
void print_arr_list(const struct arr_list *arr) {
    int i, t;
    printf("size=%d,index=%d\n", arr->size, arr->index);
    for (i = 0; i < arr->index; i++) {
        list_get(arr, i, &t);
        printf("list[%d]=%d\n", i, t);
    }
}

// 删除指定位置的数据，O(N),
// 删除数据后，所有数据向左移动
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

// 移除并返回末尾的数据, O(1)
int list_pop(struct arr_list *arr) {
    return list_removeat(arr, arr->index - 1);
}

// 判断 array list里是否包含某个数据, O(N)
int list_index(const struct arr_list *arr, int obj) {
    int i;
    for (i = 0; i < arr->index; i++) {
        if (arr->arr[i] == obj) {
            return i;
        }
    }
    return -1;
}

// 删除某个数据项，O(N), 只删第一次出现的位置，
// 删除后所有数据向左移动
int list_remove(struct arr_list *arr, int obj) {
    int i, index;
    index = list_index(arr, obj);
    if (index != -1) {
        for (i = index; i < arr->index - 1; i++) {
            arr->arr[i] = arr->arr[i + 1];
        }
        arr->index--;
    }
    return index;
}

int main(void)
{
    struct arr_list *arr;
    int r;

    arr = create_arr_list(3);
    printf("list push: 5, 6, 7\n");
    list_push(arr, 5);
    list_push(arr, 6);
    list_push(arr, 7);
    print_arr_list(arr);

    printf("list push: 8, will auto expand\n");
    list_push(arr, 8);
    print_arr_list(arr);

    printf("list remove at 2\n");
    list_removeat(arr, 2);
    print_arr_list(arr);

    printf("list pop \n");
    list_pop(arr);
    print_arr_list(arr);

    printf("list insert 0\n");
    list_insert(arr, 0, 3);
    print_arr_list(arr);

    r = list_index(arr, 3);
    printf("list index 3:%d\n", r);
    r = list_index(arr, 7);
    printf("list index 7:%d\n", r);

    printf("list remove 3\n");
    list_remove(arr, 3);
    print_arr_list(arr);

    printf("list set index 0 = 3\n");
    list_set(arr, 0, 3);
    print_arr_list(arr);

    free_arr_list(arr);
    return 0;
}

### 小结

自己用C实现一下高级语言的数据结构，便于对高级语言自带的数据结构有更深的理解。
