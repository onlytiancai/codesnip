#include <stdio.h>
#include <stdlib.h>

// 函数练习
static void hello_world(){
    printf("hello, world\n");
}

// for循环练习, 打印n次hello world
static void n_hello_world(int n){
    int i = 0;
    for (i = 0; i < n; i++) {
        printf("hello, world\n");
    }
}

// 字符串练习, 计算字符串长度
static int w_strlen(const char* str){
    int i;
    // 向后滚动指针，同时递增i，直到找到字符串结尾
    for (i = 0; *str != '\0'; str++, i++) {
        ;
    }
    return i;
}

// 理解字符串的存储
static void change_str_test(){
    // 常量不能修改
    // char* s1 = "hello"; // will core dump
    char s1[10] = "hello";
    *s1 = 'p';
    printf("%s\n", s1);
}

// 指针练习, 反转字符串
static char* reverse(char* str){
    char* ret = str;
    // 滚到字符数组末尾的\0之前
    char* p = str + w_strlen(str) - 1;
    char c;
  
    // 两个指针，一个从前往后滚，一个从后往前滚，直到就要交错之前
    // 滚动的过程中交换两个指针指向的字符
    for ( ; p > str; --p, ++str) { 
        printf("debug[reverse]: %p %p %c %c\n", p, str, *p, *str);
        c = *p;
        *p = *str;
        *str = c;
    }

    return ret;
}

// 内存申请相关，连接两个字符串
static void concat_test(){
    char* a = "hello";
    char* b = "world";
    //结果字符串长度为两个字符窜长度加\0的位置
    int len = w_strlen(a) + w_strlen(b) + 1;
    // 动态分配内存
    char* p = (char *)malloc(sizeof(char) * len);
    char* result; 

    // 必须判断是否分配到内存
    if (p != NULL){
        // 保存动态分配内存的开始指针，free时必须从这里free
        result = p;

        //滚动p和a，直到a的末尾
        while (*a != '\0') {
            printf("debug[concat_test]:while a %p %c\n", a, *a);
            *p++ = *a++;
        }

        //滚动p和b，直到b的末尾
        while (*b != '\0') {
            printf("debug[concat_test]:while b %p %c\n", a, *a);
            *p++ = *b++;
        }

        // 末尾整个0
        *p= '\0';
        printf("concat_test: %s\n", result);

        //释放动态分配的内存
        free(result);
    }else{
        printf("malloc error"); 
    }

}

//结构练习，人员统计系统
struct customer {
    char* name;
    int age;
};

static void customer_manager() {
    // 直接在栈上分配结构体
    struct customer wawa;
    struct customer* p_wawa;
    struct customer* p_customers;
    int n = 2;

    char name[] = "wawa";
    // char* name = "wawa"; //splint warning
    char name2[] = "tiancai";
   
    // 直接用结构名访问成员 
    wawa.name = name;
    wawa.age = 30;
    printf("%s is %d years old\n", wawa.name, wawa.age);

    // 用指针访问结构成员
    p_wawa = &wawa;
    p_wawa->age = 31;
    printf("%s is %d years old\n", wawa.name, wawa.age);

    // 为员工数组动态分配内存
    p_customers = (struct customer*)malloc(sizeof(struct customer) * n);
    if (p_customers != NULL) {
        // 设置数组第一项
        p_customers->name = name;
        p_customers->age = 10;

        // 设置数组第二项
        p_customers++;
        p_customers->name = name2;
        p_customers->age = 30;

        // 滚动数组外面，然后反向循环到数组开始
        p_customers++;
        while(n-- > 0){
            p_customers--;
            printf("%s is %d years old\n", p_customers->name, p_customers->age);
        }

        // 释放动态分配的内存，这时候p_customers已经位于起始位置了
        // 结构体里的name1, name2是在栈上分配的，不用释放
        free(p_customers);
    }
}

// 函数指针练习, 排序

// 正序排序的比较函数
static int cmp_default(int a, int b){
    return a - b;
}

// 反序排序的比较函数
static int cmp_reverse(int a, int b){
    return b - a; 
}

// int类型的冒泡排序算法，可传入一个比较函数指针
// 类似回调函数，该函数需要两个int参数且返回int
static void sort(int* arr, int n, int (*cmp)(int, int)){
    int i, j, t;
    int *p, *q;

    p = arr;

    for (i = 0; i < n; i++, p++) {
        q = p;
        for (j = i; j < n; j++, q++) {
            // 调用函数指针指向的函数和使用函数一样，貌似是简单写法
            if (cmp(*p, *q) > 0) {
                t = *p;
                *p = *q;
                *q = t;
            }
        }
    }
}

// 测试排序函数
static void sort_test(){
    int arr[] = {4, 5, 3, 1, 2};
    int i, n = 5;

    // 正向排序， 传入cmp_default函数的地址，貌似不需要&取地址
    sort(arr, 5, cmp_default);
    for (i = 0; i < n; i ++) {
        printf("%d%s", arr[i], i == n - 1 ? "" : ", "); 
    }
    printf("\n");

    //反向排序，同上
    sort(arr, 5, cmp_reverse);
    for (i = 0; i < n; i ++) {
        printf("%d%s", arr[i], i == n - 1 ? "" : ", "); 
    }
    printf("\n");
}


int main(){
    char str[10] = "hello";
    hello_world();

    n_hello_world(5);

    printf("len=%d\n", w_strlen("hello world"));

    change_str_test();

    printf("222 %s\n", reverse(str));

    concat_test();

    customer_manager();

    sort_test();

    return 0;
}
