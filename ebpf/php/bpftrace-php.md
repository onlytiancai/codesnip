
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
/ pid == 1416081 /
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


bpftrace -e '
uprobe:/usr/bin/php8.3:zend_execute
/ pid == 1416081 /
{
    $execute_data = (uint64)arg0;
    $func = *(uint64*)($execute_data + 0x18);
    $fname = *(uint64*)($func + 0x08);

    if ($fname != 0) {
        $str = $fname + 0x18;
        printf("method: %s\n", str($str));
    }
}
'

从 PHP 8.0 开始，Zend VM 做了非常激进的优化：

- 真实执行入口变成了execute_ex() / ZEND_VM_EXECUTE_EX()
- zend_execute() 只是一个 冷路径 wrapper
- 绝大多数用户代码 根本不会走到 zend_execute


readelf -Ws /usr/bin/php8.3 | grep execute



# bpftrace -e '
uprobe:/usr/bin/php8.3:execute_ex
/ pid == 3055002 /
{
    printf("PID %d hit execute_ex\n", pid);
}
'
Attaching 1 probe...
PID 3055002 hit execute_ex
PID 3055002 hit execute_ex


bpftrace -e '
uprobe:/usr/bin/php8.3:zend_execute
/ pid == 3055002 /
{
    printf("PID %d hit zend_execute\n", pid);
}
'

bpftrace -e '
uprobe:/usr/bin/php8.3:execute_ex
/ pid == 3055002 /
{
    $ed = (uint64)arg0;
    $func = *(uint64*)($ed + 0x18);

    if ($func == 0) {
        return;
    }

    $fname = *(uint64*)($func + 0x08);
    if ($fname == 0) {
        return;
    }

    $str = $fname + 0x18;
    printf("func: %s\n", str($str));
}
'

几个非常重要的注意点（实战经验）

1. **一定要先判空**
   bpftrace 不会帮你兜底，空指针直接炸。

2. `{main}` / include / eval
   有些 execute_data 没有 `function_name`，这是正常的。

3. ZTS vs non-ZTS
   如果你发现输出乱码或直接 crash，第一件事就是怀疑 offset。

4. 性能
   `execute_ex` 是极热路径，真实使用要加采样或过滤。

最后一句话总结你现在的位置

* 你已经找到了 **唯一正确的 hook 点：`execute_ex`**
* 方法名来自 `execute_data->func->common.function_name`
* bpftrace 场景下没有 API，**只有结构体解引用**
* 接下来你已经可以做 profiler / tracer / APM 级别的事情了


bpftrace -e '
uprobe:/usr/bin/php8.3:execute_ex
/ pid == 3055002 /
{
    $ed = (uint64)arg0;

    $func = *(uint64*)($ed + 0x18);
    if ($func == 0) { return; }

    $fname = *(uint64*)($func + 0x08);
    if ($fname == 0) { return; }

    // 先尝试从 $this 取运行期类
    $this_zv = $ed + 0x28;
    $this_val = *(uint64*)($this_zv + 0x08);

    if ($this_val != 0) {
        $obj = $this_val;
        $ce = *(uint64*)($obj + 0x00);
        $cname = *(uint64*)($ce + 0x00);

        printf("%s::%s\n",
            str($cname + 0x18),
            str($fname + 0x18)
        );
        return;
    }

    // fallback：定义作用域
    $scope = *(uint64*)($func + 0x20);
    if ($scope != 0) {
        $cname = *(uint64*)($scope + 0x00);
        printf("%s::%s\n",
            str($cname + 0x18),
            str($fname + 0x18)
        );
    } else {
        printf("%s\n", str($fname + 0x18));
    }
}
'

===

bpftrace -e '
struct zend_class_entry_min {
    u64 type;
    void *name;
};
uretprobe:/usr/bin/php8.3:zend_get_executed_scope
/ pid == 1416081 /
{
    if (retval == 0) {
        printf("retval == 0");
        return;
    }

    $ce = (struct zend_class_entry_min *)retval;
    // printf("$ce ===:%r\n", buf($ce, 64));
    if ($ce->name != 0) {

        $zs = (uint64)$ce->name;
        // printf("raw: %r\n", buf($zs, 64));

        $len = *(uint64*)($zs + 0x10);
        $str = str($zs + 0x18, $len);

        printf("class: len=%d, val=%s\n", $len, $str);

    }
}

uprobe:/usr/bin/php8.3:execute_ex
/ pid == 1416081 /
{
    $ed = (uint64)arg0;
    $func = *(uint64*)($ed + 0x18);

    if ($func == 0) {
        return;
    }

    $fname = *(uint64*)($func + 0x08);
    if ($fname == 0) {
        return;
    }

    $str = $fname + 0x18;
    printf("func: %s\n", str($str));
}
'

类名统计

bpftrace -e'
struct zend_class_entry_min {
    u64 type;
    void *name;
};

uretprobe:/usr/bin/php8.3:zend_get_executed_scope
/ pid == 1481210 /
{
    if (retval == 0) {
        return;
    }

    $ce = (struct zend_class_entry_min *)retval;
    if ($ce->name == 0) {
        return;
    }

    $zs = (uint64)$ce->name;
    $len = *(uint64*)($zs + 0x10);

    // 简单防御，避免异常长度
    if ($len <= 0 || $len > 256) {
        return;
    }

    $class = str($zs + 0x18, $len);
    @classes[$class] = count();
}

interval:s:5
{
    printf("\n=== Class statistics (5s) ===\n");
    print(@classes);
    clear(@classes);
    exit();
}

'|sort