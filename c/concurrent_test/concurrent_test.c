#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <pthread.h>

#define N 1000000 //定义求和数组的长度
#define cpu_count 4 //定义CPU个数

static int array[N]; //被求和的数组
static int partial_results[cpu_count]; //保存各线程的分部结果

//随机初始化一个被求和数组
void init_array() {
    int i;

    srand((unsigned)time(NULL));
    for (i = 0; i < N; i++) {
        array[i] = rand() % (100 - 1) + 1;
    }
}

//重置各线程的分部结果
void reset_partial_results(){
    int i;
    for (i = 0; i < cpu_count; i++) {
        partial_results[i] = 0;
    }
}

//合并各线程结果
int merge_partial_result(){
    int i, result = 0;
    for (i = 0; i < cpu_count; i++) {
        result += partial_results[i];
    }
    return result;
}

//多线程执行一个函数，传入的函数指针需指向一个**接受一个int指针参数并返回void的函数**, 参数将会是cpu_index
void parallel_run(void (*proc)(int*)){
    pthread_t tid[cpu_count];
    int cpu_index, ret, indexes[cpu_count];
    for (cpu_index = 0; cpu_index < cpu_count; cpu_index++) {
        indexes[cpu_index] = cpu_index;

        ret=pthread_create(&tid[cpu_index], NULL, (void *)proc, &indexes[cpu_index]);
        if(ret!=0){
            printf ("Create pthread error!\n");
            exit (1);
        }

    }

    for (cpu_index = 0; cpu_index < cpu_count; cpu_index++) {
          pthread_join(tid[cpu_index], NULL);
    }

}

//串行求和算法，当作单元测试中的期望值
int sum() {
    int i, result = 0;
    for (i = 0; i < N ; i++) {
        result += array[i];
    }
    return result;
}

//并行求和算法1中的线程执行函数
void sum1_proc(int* pcpu_index){
    int cpu_index = *pcpu_index;

    //得到要处理的数据边界
    int len = (N - 1) / cpu_count + 1;
    int begin, end;
    begin = len * cpu_index;
    end = begin + len;
    if (end > N) { end = N; }

    int i;
    for (i = begin; i < end; i++) {
        //多个线程间频繁交替写全局变量partial_results，造成cacheline失效
        partial_results[cpu_index] += array[i];
    }
}

//并行求和版本1
int sum1(){
    reset_partial_results();
    parallel_run(&sum1_proc); 
    return merge_partial_result();
}

//并行求和版本2中的线程执行函数
void sum2_proc(int* pcpu_index){
    int cpu_index = *pcpu_index;

    int len = (N - 1) / cpu_count + 1;
    int begin, end;
    begin = len * cpu_index;
    end = begin + len;
    if (end > N) { end = N; }

    int i, temp_result = 0;
    for (i = begin; i < end; i++) {
        temp_result += array[i];
    }
    //计算完毕后写一次全局变量partial_results，性能提高很多
    partial_results[cpu_index] = temp_result;
}

//并行求和版本2
int sum2(){
    reset_partial_results();
    parallel_run(&sum2_proc); 
    return merge_partial_result();
}


//单元测试
void unit_test(){
    int expected = sum();
    int result = 0; 

    printf("expected=%d\n", expected);

    result = sum1(); 
    printf("sum1 result=%d, test %s\n", result, result == expected ? "Ok": "faild");


    result = sum2(); 
    printf("sum2 result=%d, test %s\n", result, result == expected ? "Ok": "faild");
}

int main(int argc, const char *argv[]) {
    if (argc < 2 ) {
        printf("usage:./concurrent_test.o sum 1000\n");
        printf("usage:./concurrent_test.o test\n");
        return 1;
    }
    
    const char* action = argv[1];
    int iter_count = argc == 3 ? atoi(argv[2]) : 1;

    init_array();

    // 单元测试
    if (strcmp(action, "test") == 0) {
        unit_test();       
        return 0;
    }

    // 根据参数选择不同的求和算法
    int (*sumfunc)();
    if (strcmp(action, "sum") == 0) {
        sumfunc = &sum;
    }
    else if (strcmp(action, "sum1") == 0) {
        sumfunc = &sum1;
    }
    else if (strcmp(action, "sum2") == 0) {
        sumfunc = &sum2;
    }else{
        sumfunc = &sum;
    }


    int i;
    for (i = 0; i < iter_count; i++) {
        (*sumfunc)();
    }
    printf("%s excute %d.\n", action, iter_count);
    return 0;
}
