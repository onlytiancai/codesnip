#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>

#define N 1000000 //定义求和数组的长度
#define cpu_count 4 //定义CPU个数

static int array[N]; //被求和的数组
static int sum1_results[cpu_count]; //保存sum1并行求和算法的临时结果

//随机初始化一个被求和数组
void init_array() {
    int i;

    srand((unsigned)time(NULL));
    for (i = 0; i < N; i++) {
        array[i] = rand() % (100 - 1) + 1;
    }
}

//sum1算法中每个线程求一部分数组元素的和
void sum1_partial_sum(int* pcpu_index){
    int cpu_index = *pcpu_index;

    int len = (N - 1) / cpu_count + 1;
    int begin, end;
    begin = len * cpu_index;
    end = begin + len;
    if (end > N) { end = N; }

    int i;
    for (i = begin; i < end; i++) {
        sum1_results[cpu_index] += array[i];
    }
}

//sum1算法的合并各线程结果
int sum1_merge_partial_result(){
    int i, result = 0;
    for (i = 0; i < cpu_count; i++) {
        result += sum1_results[i];
    }
    return result;
}

//sum1算法，起cpu_count个线程，每个线程求数组的一部分的和，最后汇总各线程结果
int sum1(){
    pthread_t tid[cpu_count];
    int cpu_index, ret, indexes[cpu_count];
    for (cpu_index = 0; cpu_index < cpu_count; cpu_index++) {
        indexes[cpu_index] = cpu_index;

        ret=pthread_create(&tid[cpu_index], NULL, (void *)sum1_partial_sum, &indexes[cpu_index]);
        if(ret!=0){
            printf ("Create pthread error!\n");
            exit (1);
        }

    }

    for (cpu_index = 0; cpu_index < cpu_count; cpu_index++) {
          pthread_join(tid[cpu_index], NULL);
    }

    return sum1_merge_partial_result();
}

//串行求和算法，当作单元测试中的期望值
int sum() {
    int i, result = 0;
    for (i = 0; i < N ; i++) {
        result += array[i];
    }
    return result;
}

//单元测试
int unit_test(){
    int expected = sum();
    int result = 0; 

    printf("expected=%d\n", expected);

    result = sum1(); 
    printf("sum1 result=%d, test %s\n", result, result == expected ? "Ok": "faild");
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
