## 快速学习C语言一: 造轮子，ArrayList

高级语言里的列表是最常用的数据结构，在C里造个轮子玩玩，C没有泛型，先用int练习。

Collection的ADT一般有hasnext,next，add, remove操作，List一般还加了removeat,
insert等，然后Stack有push和pop，Queue有enqueue和dequeue。列表有种实现，
ArrayList和LinkedList，总体来说ArrayList更常用一些，就先用数组实现个列表。

ArrayList在末尾的添加和删除还是挺快的(O(1))，所以当栈来用挺好，Push和Pop都在末尾。
当队列的话，Enqueue在数组头部放入元素，所有右边的元素都要向右移动(O(N))，比较耗
时，不如LinkedList。另外获取指定索引数据或设置指定索引的数据是O(1)复杂度，
比LinkedList快，如果要想查看是否有指定数据，或删除指定数据，要扫描全表，复杂度
是O(N)。哦，删除指定位置的数据复杂度也是O(N), 因为删除后右边的数据要全部向
左移动。

可以开始了，首先定义一个结构来保存状态

    struct arr_list {
        int * arr;      // 内部数组
        int index;      // 实际数据大小
        int size;       // 预分配空间大小
    };

创建一个array list

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

空间不足时自动扩容，默认策略是空间不够时申请双倍大小空间, 然后把原有数据拷贝到
新空间，并把原有空间释放掉, 该函数一般是新增元素前调用，所以判断条件是当实际
所用空间已经等于或大于(应该不可能)预分配空间时扩容。

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

在指定位置插入新元素，现有元素向右移，O(N)

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

在array list 末尾插入数据， O(1)

    int list_push(struct arr_list *arr, int obj) {
        return list_insert(arr, arr->index, obj);
    }

删除指定位置的数据，O(N),  删除数据后，所有数据向左移动

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

移除并返回末尾的数据, O(1)

    int list_pop(struct arr_list *arr) {
        return list_removeat(arr, arr->index - 1);
    }

判断 array list里是否包含某个数据, O(N)

    int list_index(const struct arr_list *arr, int obj) {
        int i;
        for (i = 0; i < arr->index; i++) {
            if (arr->arr[i] == obj) {
                return i;
            }
        }
        return -1;
    }

删除某个数据项，O(N), 只删第一次出现的位置， 删除后所有数据向左移动

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


释放一个array list的内存

    int free_arr_list(struct arr_list *arr){
        free(arr->arr);
        free(arr);
        return 0;
    }

从头打印一个arr list

    void print_arr_list(const struct arr_list *arr) {
        int i, t;
        printf("size=%d,index=%d\n", arr->size, arr->index);
        for (i = 0; i < arr->index; i++) {
            list_get(arr, i, &t);
            printf("list[%d]=%d\n", i, t);
        }
    }

最后整体测试一下

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

用C实现一下高级语言里常用的数据结构，可以对它们有更深的理解。
