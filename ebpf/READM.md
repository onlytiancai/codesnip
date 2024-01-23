使用
    gcc test.c -o test.o
    objdump -tT test.o 
    sudo bpftrace -e 'uprobe:./test.o:foo{printf("begin foo: arg 0=%d\n", arg0);@start = nsecs;}uretprobe:./test.o:foo{printf("end foo: retval=%d, cost=%ld ns\n", retval, nsecs-@start);}END{printf("program end\n");clear(@start);}' 
