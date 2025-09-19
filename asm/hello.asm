section .data
    msg db 'Hello, World!', 0xa   ; 定义要打印的字符串，0xa是换行符
    len equ $ - msg             ; 计算字符串的长度

section .text
    global _start               ; 程序的入口点，必须是全局的

_start:
    ; write 系统调用
    ; rax = 1 (sys_write)
    ; rdi = 1 (stdout)
    ; rsi = 地址 (msg)
    ; rdx = 长度 (len)
    mov rax, 1
    mov rdi, 1
    mov rsi, msg
    mov rdx, len
    syscall

    ; exit 系统调用
    ; rax = 60 (sys_exit)
    ; rdi = 0 (退出状态码)
    mov rax, 60
    mov rdi, 0
    syscall
