### 测试各种并行求和算法

编译和运行

```
make  //编译
make test  //单元测试
make profiler //进行性能剖析

./concurrent_test.o sum 1000 //串行求和迭代1000次
./concurrent_test.o sum1 1000 //并行求和迭代1000次
./concurrent_test.o sum2 1000 //并行求和迭代1000次

```

四核机器上执行结果如下
```
$ make test //单元测试，全部通过
./concurrent_test.o test
expected=49959687
sum1 result=49959687, test Ok
sum2 result=49959687, test Ok

$ time ./concurrent_test.o sum 100    //串行求和：228毫秒
sum excute 100.

real    0m0.228s
user    0m0.220s
sys 0m0.000s

$ time ./concurrent_test.o sum1 100   //并行求和版本1：389毫秒
sum1 excute 100.

real    0m0.389s
user    0m1.350s
sys 0m0.000s

$ time ./concurrent_test.o sum2 100   //并行求和版本2: 86毫秒
sum2 excute 100.

real    0m0.086s
user    0m0.240s
sys 0m0.010s

```
