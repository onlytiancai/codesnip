
查看符号


# bpftrace -l 'uprobe:/usr/bin/php8.3:*zend*' | grep execute
uprobe:/usr/bin/php8.3:zend_execute
uprobe:/usr/bin/php8.3:zend_execute_scripts
uprobe:/usr/bin/php8.3:zend_get_executed_filename
uprobe:/usr/bin/php8.3:zend_get_executed_filename_ex
uprobe:/usr/bin/php8.3:zend_get_executed_lineno
uprobe:/usr/bin/php8.3:zend_get_executed_scope
uprobe:/usr/bin/php8.3:zend_init_code_execute_data
uprobe:/usr/bin/php8.3:zend_init_execute_data
uprobe:/usr/bin/php8.3:zend_init_func_execute_data


先验证 zend_execute 是否可用，确认 probe 能命中


bpftrace -e '
uprobe:/usr/bin/php8.3:zend_execute
{
    printf("PID %d hit zend_execute\n", pid);
}
interval:s:1{
    exit();
}
'

bpftrace -e '
uretprobe:/usr/bin/php8.3:zend_get_executed_scope
{
    if (retval == 0) {
        return;
    }

    printf("PID %d scope_ptr=%p\n", pid, retval);
    printf("%r\n", buf(retval, 64));
}
interval:s:1{
    exit();
}
'


打印文件名


BPFTRACE_STRLEN=200 bpftrace -e '
uretprobe:/usr/bin/php8.3:zend_get_executed_filename
{
    if (retval) {
        printf("PID %d PHP file: %s\n", pid, str(retval));
    }
}
uretprobe:/usr/bin/php8.3:zend_get_executed_lineno
{
    if (retval) {
        printf("PID %d PHP line: %d\n", pid, retval);
    }
}
'

打印行号

BPFTRACE_STRLEN=200 bpftrace -e '
uretprobe:/usr/bin/php8.3:zend_get_executed_lineno
{
    if (retval) {
        printf("PID %d line: %d\n", pid, retval);
    }
}'


过滤 PID


BPFTRACE_STRLEN=200 bpftrace -e '
uretprobe:/usr/bin/php8.3:zend_get_executed_filename
/ pid == 1978752 /
{
    if (retval) {
        printf("PID %d PHP file: %s\n", pid, str(retval));
    }
}'






===
export BPFTRACE_STRLEN=64

BPFTRACE_STRLEN=200 bpftrace -e '
struct zend_string {
    u64 gc;
    u64 h;
    u64 len;
    char val[1];
};

struct zend_function_common {
    void *type;
    void *function_name;   // zend_string *
    void *scope;           // zend_class_entry *
};

struct zend_function {
    struct zend_function_common common;
};

struct zend_execute_data {
    void *opline;
    struct zend_function *func;
};

uprobe:/usr/bin/php8.3:zend_execute
{
    $ex = (struct zend_execute_data *)arg0;
    $func = $ex->func;

    if ($func && $func->common.function_name) {
        $zs = (struct zend_string *)$func->common.function_name;
        printf("PID %d func=%s\n", pid, str($zs->val));
    }
}
'

===

gdb -p 523800 -batch -nx -q   -ex "bt"   -ex "detach"   -ex "quit"


#0  0x0000f8d28f9f49e0 in __GI___clock_nanosleep (clock_id=<optimized out>, clock_id@entry=0, flags=flags@entry=0, req=req@entry=0xffffd62976e8, rem=rem@entry=0xffffd62976e8) at ../sysdeps/unix/sysv/linux/clock_nanosleep.c:78
#1  0x0000f8d28f9f9ccc in __GI___nanosleep (req=req@entry=0xffffd62976e8, rem=rem@entry=0xffffd62976e8) at ../sysdeps/unix/sysv/linux/nanosleep.c:25
#2  0x0000f8d28f9f9b8c in __sleep (seconds=0) at ../sysdeps/posix/sleep.c:55
#3  0x0000ab10b7ae20e0 in ?? ()
#4  0x0000ab10b7c42f30 in execute_ex ()
#5  0x0000ab10b7c4721c in zend_execute ()
#6  0x0000ab10b7bc22f4 in zend_execute_scripts ()
#7  0x0000ab10b7b546e0 in php_execute_script ()
#8  0x0000ab10b7cbe58c in ?? ()
#9  0x0000ab10b79dee2c in ?? ()
#10 0x0000f8d28f967400 in __libc_start_call_main (main=main@entry=0xab10b79deb60, argc=argc@entry=15, argv=argv@entry=0xffffd629b3a8) at ../sysdeps/nptl/libc_start_call_main.h:58
#11 0x0000f8d28f9674d8 in __libc_start_main_impl (main=0xab10b79deb60, argc=15, argv=0xffffd629b3a8, init=<optimized out>, fini=<optimized out>, rtld_fini=<optimized out>, stack_end=<optimized out>) at ../csu/libc-start.c:392
#12 0x0000ab10b79def30 in _start ()
[Inferior 1 (process 523800) detached]


找出 etime 最大的 php 进程

ps -eo pid=,etime=,cmd= | grep php | sort -t- -k1,1nr -k2,2nr

bpftrace -e '
uretprobe:/usr/bin/php8.3:zend_get_executed_scope
/ pid == 1641373 /
{
    if (retval == 0) {
        return;
    }

    printf("PID %d scope_ptr=%p\n", pid, retval);
    printf("%r\n", buf(retval, 64));
}
'

retval ===:\x02\x00\x00\x00\x00\x00\x00\x00@92\x85\xe0\xea\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00H\x13\x04\x00\x02\x00\x00\x00\x00\x00\x00\x00@)\xbf\x85\xe0\xea\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00
$ce ===:\x02\x00\x00\x00\x00\x00\x00\x00@92\x85\xe0\xea\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00H\x13\x04\x00\x02\x00\x00\x00\x00\x00\x00\x00@)\xbf\x85\xe0\xea\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00
$zs ===:\x89\x96\x00\x00v\x00\x00\x00\x12!u\xb4Q\x110\xa8'\x00\x00\x00\x00\x00\x00\x00Symfony\Component\Console\Output\Output\x00
class =