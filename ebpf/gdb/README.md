用 gdb 查看 struct 定义

编译

    $ gcc -g test.c -o test.o
    $ ./test.o
    hello 1024 2048

gdb 调试

    $ gdb -q test.o
    Reading symbols from test.o...
    (gdb) set print pretty on
    (gdb) b print_data
    Breakpoint 1 at 0x1169: file test.c, line 7.
    (gdb) r
    Starting program: /home/ubuntu/src/codesnip/ebpf/gdb/test.o

    Breakpoint 1, print_data (a=0x7fffffffe5e7) at test.c:7
    7       void print_data(struct AAA *a) {
    (gdb) n
    8           printf("%s %d %d\n", a->s, a->b.x,a->b.y);
    (gdb) p a
    $1 = (struct AAA *) 0x7fffffffe5e0
    (gdb) p *a
    $2 = {
      s = 0x55555555600e "hello",
      b = {
        x = 1024,
        y = 2048
      }
    }
    (gdb)

bpftrace 跟踪

    bpftrace -e 'uprobe:./test.o:print_data{printf("print_data:%d\n", arg0)}';
