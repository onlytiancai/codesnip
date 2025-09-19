在Linux上使用汇编语言编写“Hello, World\!”，需要根据你使用的汇编器和目标架构选择不同的语法。最常见的是使用 **NASM (Netwide Assembler)** 汇编器，它语法简洁且跨平台。

下面我们将展示一个使用NASM汇编器，在 **64位 (x86-64)** Linux系统上的例子。

### Hello, World\! (64位 Linux)

这个程序会调用Linux内核的系统调用来打印字符串并退出。

```nasm
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
```

-----

### 编译和运行

将上面的代码保存为 `hello.asm` 文件，然后按照以下步骤进行编译和运行：

1.  **安装NASM**（如果尚未安装）：
    在基于Debian/Ubuntu的系统上：

    ```sh
    sudo apt-get install nasm
    ```

    在基于Fedora/CentOS的系统上：

    ```sh
    sudo dnf install nasm
    ```

2.  **汇编和链接**：
    首先，使用NASM将汇编代码编译成目标文件（.o）：

    ```sh
    nasm -f elf64 hello.asm -o hello.o
    ```

    `-f elf64` 指定了输出格式为64位ELF（Executable and Linkable Format），这是Linux的标准可执行文件格式。

    接下来，使用ld（GNU链接器）将目标文件链接成可执行文件：

    ```sh
    ld hello.o -o hello
    ```

    `ld` 命令将 `hello.o` 链接成名为 `hello` 的可执行文件。

3.  **运行程序**：

    ```sh
    ./hello
    ```

你应该能在终端看到输出：

```
Hello, World!
```
