gdb test

    gcc -g test2.c -o test.o
    gdb test.o
    (gdb) b 1
    (gdb) r
    Breakpoint 1, main (argc=0, argv=0x7ffff7ddbea0) at test2.c:7
    (gdb) n
    9           data.b = 100;
    (gdb) n
    10          printf("%p %p %ld\n", &data, &(data.b), (long)&(data.b)-(long)&data);
    (gdb) ptype /o test_data
    No symbol "test_data" in current context.
    (gdb) ptype /o struct test_data
    /* offset    |  size */  type = struct test_data {
    /*    0      |  1272 */    char a[1272];
    /* 1272      |     1 */    char b;

                               /* total size (bytes): 1273 */
                             }

bpftrace

    bpftrace -e 'struct data {char a[1272];char b;};uprobe:./test.o:print_data{printf("print_data:%d\n", ((struct data*)(arg0))->b);}'
