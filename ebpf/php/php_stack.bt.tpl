/* CPU 采样 */
profile:hz:97
/pid == __PID__/
{
    @cpu[kstack, ustack] = count();
}

/* off-CPU */
tracepoint:sched:sched_switch
/args->prev_pid == __PID__/
{
    @offcpu[kstack, ustack] = count();
}

/* sleep */
tracepoint:syscalls:sys_enter_nanosleep
/pid == __PID__/
{
    @sleep[ustack] = count();
}

/* IO wait */
tracepoint:syscalls:sys_enter_epoll_pwait,
tracepoint:syscalls:sys_enter_epoll_pwait2,
tracepoint:syscalls:sys_enter_ppoll
/pid == __PID__/
{
    @io_wait[ustack] = count();
}

/* 到时间直接输出并退出 */
interval:s:__DURATION__
{
    printf("\n=== CPU STACKS ===\n");
    print(@cpu);

    printf("\n=== OFF-CPU STACKS ===\n");
    print(@offcpu);

    printf("\n=== SLEEP STACKS ===\n");
    print(@sleep);

    printf("\n=== IO WAIT STACKS ===\n");
    print(@io_wait);

    exit();
}