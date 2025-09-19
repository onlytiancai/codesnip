section .data
    result_msg db "Result: ", 0
    result_len equ $ - result_msg
    newline db 0xA 

section .bss
    result_str resb 32    
    
section .text
    global _start
    
_start:
    ;--- 获取命令行参数 ---
    mov r8, [rsp]       ; r8 = argc
    cmp r8, 3
    jne exit_error

    ;--- 将第一个参数转换为整数 ---
    mov rdi, [rsp + 16] 
    call str_to_int     
    mov r9, rax         ; 保存第一个数

    ;--- 将第二个参数转换为整数 ---
    mov rdi, [rsp + 24] 
    call str_to_int
    add r9, rax         ; r9 += 第二个数

    ; 打印 "Result: "
    mov rax, 1
    mov rdi, 1
    mov rsi, result_msg
    mov rdx, result_len
    syscall

    ;--- 将结果转换为字符串 ---
    mov rdi, r9         ; 将结果放入 rdi
    mov rsi, result_str ; 缓冲区地址
    mov rdx, 32         ; 缓冲区大小
    call int_to_str     
    
    ; 打印结果字符串
    ; int_to_str 返回字符串的地址在 rax，长度在 rdx
    mov rsi, rax        ; 将地址移到 rsi
    mov rdx, rdx        ; 保持长度在 rdx
    mov rax, 1
    mov rdi, 1
    syscall

    ; 打印换行符
    mov rax, 1
    mov rdi, 1
    mov rsi, newline
    mov rdx, 1
    syscall

    ;--- 正常退出 ---
    mov rax, 60
    xor rdi, rdi
    syscall

exit_error:
    mov rax, 60
    mov rdi, 1
    syscall

;--- 子程序: 字符串转整数 ---
; rdi = 字符串地址
; rax = 转换后的整数
str_to_int:
    xor rax, rax         
    xor rcx, rcx         
.loop:
    movzx rbx, byte [rdi + rcx]  
    cmp rbx, '0'
    jl .end                      
    cmp rbx, '9'
    jg .end
    
    sub rbx, '0'                 
    imul rax, 10                 
    add rax, rbx                 
    inc rcx
    jmp .loop
.end:
    ret

;--- 子程序: 整数转字符串 ---
; rdi = 整数
; rsi = 缓冲区地址
; rdx = 缓冲区大小
; rax = 字符串地址
; rdx = 字符串长度
int_to_str:
    mov r8, rsi          ; 缓冲区起始地址
    mov r9, rdx          ; 缓冲区大小
    mov r10, rdi         ; 要转换的整数
    
    xor rcx, rcx         ; 计数器
    mov rbx, 10          ; 除数

.loop:
    xor rdx, rdx         
    mov rax, r10         ; 将要除的数放入 rax
    div rbx              ; rax = r10 / 10, rdx = r10 % 10
    mov r10, rax         ; 更新要除的数
    
    add rdx, '0'         ; 余数转为 ASCII
    mov byte [r8+rcx], dl 
    
    inc rcx              
    cmp r10, 0           
    jne .loop

    ; 反转字符串，使之正序
    mov rsi, r8          
    mov rdi, r8          
    add rdi, rcx         
    dec rdi              
.reverse_loop:
    cmp rsi, rdi
    jge .done_reverse
    
    mov al, [rsi]
    mov bl, [rdi]
    mov [rsi], bl
    mov [rdi], al
    
    inc rsi
    dec rdi
    jmp .reverse_loop

.done_reverse:
    mov rax, r8          ; 返回字符串地址
    mov rdx, rcx         ; 返回字符串长度
    ret
