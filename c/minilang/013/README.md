# 第 13 步 · 完整版收官 + 模块化重构

> 进度 ████████████ 13/13 ｜ 上一节：[第 12 步 · 错误处理与诊断](../012/README.md) ｜ 下一节：[回到项目总览](../README.md)

这是整套教程的**最后一步**。前 12 步把语言功能一个个加上去，
这一步做两件事：**收官**（把学到的东西串起来，看看还差什么），
以及**重构**（把 1281 行的单文件拆成 6 个文件，并证明行为完全一致）。

## 1. 本步目标

读完你将能够：

- 用一张表说清 MiniLang 现在**完整的语言能力**。
- 解释为什么一个 1281 行的文件该拆开，以及**怎么拆**（按职责，不按行数）。
- 读懂 `minilang.h` 里的头号防重（`#ifndef`）和"模块间接口声明"。
- 看懂 `Makefile` 里的模式规则、`-lm` 为什么必须放在源文件后面。
- **证明**拆分前后行为完全一致（两个二进制跑同一套用例，逐字节对拍）。

## 2. 运行方式

本步有**两个**实现：

```sh
cd 013

# 单文件版（和第 12 步完全一样）
cc -std=c11 -Wall -Wextra -O2 minilang.c -lm -o .single
./.single demo.ml

# 多文件版
make
./minilang demo.ml

# 一键测试：两个版本都跑一遍，再逐字节对拍
./run.sh
```

## 3. 现在的程序能做什么

### 3.1 完整的能力表

13 步下来，MiniLang 现在能跑这样的程序：

```
fn factorial(n) {
    if (n <= 1) {
        return 1;
    }
    return n * factorial(n - 1);
}

let sum = 0;

for (let i = 1; i <= 5; i = i + 1) {
    sum = sum + i;
}

let result = factorial(5);
let root = sqrt(81);

print("sum = {sum}\n");
print("factorial = {result}\n");
print("sqrt = {root}\n");
```

真实输出：

```
sum = 15
factorial = 120
sqrt = 9
```

**语法层面**：

| 类别 | 语法 | 哪一步 |
|---|---|---|
| 数字 | `10`、`3.14` | 001 |
| 变量 | `let x = 1;`、`x = 2;` | 005 |
| 运算符 | `+ - * / %` | 003 |
| 比较与逻辑 | `== != < <= > >= && \|\| !` | 004 |
| 代码块 | `{ ... }` | 005 |
| 条件 | `if (c) { } else { }` | 006 |
| 循环 | `for (init; cond; update) { }` | 007 |
| 函数 | `fn f(a, b) { }` | 010 |
| 返回 | `return expr;`、`return;` | 011 |
| 字符串与模板 | `print("x = {x}\n")` | 009 |

**运行时语义**：

| 规则 | 说明 |
|---|---|
| 真值 | `0` 是假，非 `0` 是真；比较结果是 `0`/`1` |
| 短路 | `&&` / `\|\|` 的右边可能根本不求值 |
| 数字类型 | 内部全是 `double`，`10 / 4` = `2.5` |
| 输出格式 | `%.15g`，所以 `15` 打成 `15` 而不是 `15.000000` |
| 作用域 | 块级作用域，内层遮蔽外层，同作用域重复 `let` 报错 |
| 函数 | 每次调用一个新作用域；`=` 能改外层变量，`let` 不能 |
| 递归 | 深度上限 128 |
| 循环 | 步数上限 100 万 |
| 错误 | 带行列号 + 源码行 + `^` 指示符，退出码 1 |

**三种模式**（第 1 步定型，全程没变）：

```sh
./minilang demo.ml           # run：执行（默认）
./minilang demo.ml tokens    # 词法分析：打印 token 列表
./minilang demo.ml ast       # 语法分析：打印语法树
```

### 3.2 两种实现的目录结构

```
013/
├── minilang.c          ← 单文件版，1281 行，和第 12 步一字不差
├── Makefile            ← 多文件版的构建脚本
├── src/
│   ├── minilang.h      ← 公共头文件：161 行
│   ├── lexer.c         ← 词法分析 + 错误报告：267 行
│   ├── ast.c           ← AST 节点 + 打印：103 行
│   ├── parser.c        ← 语法分析：281 行
│   ├── interpreter.c   ← 求值 + 执行 + 内置函数 + 函数调用：417 行
│   └── main.c          ← 读文件 + 模式分发 + main：103 行
├── demo.ml
├── run.sh              ← 两个版本都测 + 逐字节对拍
└── tests/              ← 和第 12 步一模一样
```

## 4. 前置知识回顾

前 12 步都在**一个文件**里写代码。现在它有 1281 行了：

```text
minilang.c
├── 头注释 + include + 宏定义
├── Token 类型 + 错误处理 + 词法分析器 + type_name
├── Node 类型 + 节点池
├── 语法分析器（表达式 + 语句）
├── 打印语法树
├── 内置函数表
├── 运行时：作用域 / Value / Flow / eval / exec
├── 读文件 + 三种模式 + main
```

想改一个函数？得翻 1281 行找到它。
想确认某个全局变量还有谁在用？得全文搜索。
想单独测一下词法分析器？得先把整个文件编译一遍。

## 5. 本步引入的核心概念

### 5.1 概念一：按职责拆，不按行数拆

拆分的依据是**职责**，不是"每个文件多少行"：

```text
                    ┌──────────────┐
                    │ minilang.h   │  所有模块共用的类型、常量、函数声明
                    └──────┬───────┘
           ┌───────────┬───┴────┬──────────┬──────────┐
           │           │        │          │          │
      ┌────▼────┐ ┌────▼────┐ ┌─▼───────┐ ┌▼────────┐ ┌▼─────────┐
      │lexer.c  │ │ ast.c   │ │parser.c │ │interp-  │ │ main.c   │
      │         │ │         │ │         │ │reter.c  │ │          │
      │词法分析  │ │AST 节点 │ │语法分析  │ │求值执行  │ │入口      │
      │错误报告  │ │打印树   │ │         │ │内置函数  │ │模式分发  │
      └─────────┘ └─────────┘ └─────────┘ └─────────┘ └──────────┘
```

**依赖方向**（箭头表示"用到了"）：

```text
lexer.c  ──►  minilang.h
ast.c    ──►  minilang.h
parser.c ──►  minilang.h
interpreter.c ──►  minilang.h
main.c   ──►  minilang.h
```

**没有一个 .c 文件 include 另一个 .c 文件**——这是关键。
所有跨模块的交流都通过 `minilang.h` 里的声明完成。

**为什么不按行数拆**（比如"每 300 行一个文件"）？
因为那样切出来的文件边界可能正好把一个函数劈成两半。
按职责拆，每个文件都是"一个完整的职责"，读起来才清楚。

### 5.2 概念二：`static` 的含义变了

单文件版里到处是 `static`：

```c
static Token toks[MAX_TOKENS];
static void lex(void) { ... }
```

在 C 里，`static` 修饰一个**文件作用域**的变量或函数，意思是
"**这个翻译单元内部可见**"。单文件时，"翻译单元"就是整个程序，所以
`static` 等于"全局私有"。

拆成多文件之后，翻译单元变成了每个 `.c` 文件：

| 单文件版 | 多文件版 | 含义 |
|---|---|---|
| `static Token toks[];` | `extern Token toks[];`（在 .h）<br>`Token toks[];`（在 .c） | 变成**全局公开** |
| `static void lex(void)` | `void lex(void);`（在 .h）<br>`void lex(void)`（在 .c） | 变成**全局公开** |
| `static int find_var(...)` | 保持 `static` | 仍然**文件内私有** |

**规则很简单**：

- **只有这一个 .c 文件用**的 → 保持 `static`（比如 `find_var`、`env_push`、`bi_abs`）。
- **别的 .c 文件要用**的 → 去掉 `static`，在 `minilang.h` 里加声明。

对照一下就清楚了：

```c
/* src/lexer.c —— 别的文件要用，所以不 static */
Token       toks[MAX_TOKENS];
int         ntok;
void        lex(void);
void        error_at(int line, int col, const char *fmt, ...);

/* src/interpreter.c —— 只有自己用，保持 static */
static int  find_var(Env *e, const char *name) { ... }
static void env_push(Env *e, const char *name, double v) { ... }
```

### 5.3 概念三：头号防重

`minilang.h` 的头尾：

```c
#ifndef MINILANG_H
#define MINILANG_H

... 所有声明 ...

#endif /* MINILANG_H */
```

**为什么需要它？** 想象 `parser.c` 里同时 include 了 `minilang.h` 和 `ast.h`，
而 `ast.h` 里又 include 了 `minilang.h`——那么 `minilang.h` 的内容会被
**处理两次**，里面的类型定义就会重复，报一堆
"redefinition of 'Token'"。

`#ifndef` 的意思是"**如果这个宏还没被定义过**"：

- 第一次 include `minilang.h`：`MINILANG_H` 没定义 → 定义它 → 内容生效；
- 第二次 include：`MINILANG_H` 已定义 → 整块内容跳过。

这套东西叫 **include guard（头号防重）**，几乎所有 C 项目都会用。

（另一种写法是 `#pragma once`，更短，但严格来说不是 C 标准，
个别老编译器不支持。教学版用标准的 `#ifndef`。）

### 5.4 概念四：Makefile 的模式规则

```make
CC      ?= cc
CFLAGS  ?= -std=c11 -Wall -Wextra -O2
LDLIBS  ?= -lm

TARGET  := minilang
SRCDIR  := src
SRCS    := $(SRCDIR)/lexer.c $(SRCDIR)/ast.c $(SRCDIR)/parser.c \
           $(SRCDIR)/interpreter.c $(SRCDIR)/main.c
OBJS    := $(SRCS:.c=.o)
DEPS    := $(SRCDIR)/minilang.h

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LDLIBS)

$(SRCDIR)/%.o: $(SRCDIR)/%.c $(DEPS)
	$(CC) $(CFLAGS) -c -o $@ $<

run: $(TARGET)
	./$(TARGET) demo.ml

test: $(TARGET)
	./run.sh

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all run test clean
```

**逐行解释**：

- `CC ?= cc` —— `?=` 是"**如果还没定义才赋值**"。
  这样用户可以 `make CC=clang` 换编译器，而不用改 Makefile。
  （`=` 是无条件赋值，`:=` 是立即展开的赋值。）
- `$(SRCS:.c=.o)` —— 这是 Make 的**替换后缀**语法，
  把 `src/lexer.c` 变成 `src/lexer.o`。等价于手写五个 `src/xxx.o`。
- `$(TARGET): $(OBJS)` —— 目标依赖所有 `.o` 文件。
- `$(SRCDIR)/%.o: $(SRCDIR)/%.c $(DEPS)` —— **模式规则**：
  `%` 是通配符，表示"任意一个文件名"。
  这条规则的意思是"任何 `src/xxx.c` 都能编译成 `src/xxx.o`"。
- `$<` 是**第一个依赖**（也就是 `src/xxx.c`），`$@` 是**目标**（`src/xxx.o`）。
- `$(DEPS)` 让所有 `.o` 都依赖 `minilang.h`——
  **改了头文件，所有 `.c` 都会重新编译**。少了这一条，
  改了头文件却只重编部分文件，就会得到"链接错误"或者更诡异的行为。
- **`-lm` 必须在源文件/目标文件后面**！这是链接器的规定：
  链接器**从左到右**扫描，遇到未解析的符号时，去**后面**的库里找。
  `cc -lm foo.o` 会报 "undefined reference to `sqrt`"。
- `.PHONY: all run test clean` —— 告诉 make 这几个是**伪目标**。
  否则如果你有一个叫 `clean` 的文件，make 会以为它已经是最新的，什么都不做。
- 命令行**必须用 Tab 缩进**，不能用空格！这是 Makefile 最常见的坑。

真实 `make` 输出：

```
cc -std=c11 -Wall -Wextra -O2 -c -o src/lexer.o src/lexer.c
cc -std=c11 -Wall -Wextra -O2 -c -o src/ast.o src/ast.c
cc -std=c11 -Wall -Wextra -O2 -c -o src/parser.o src/parser.c
cc -std=c11 -Wall -Wextra -O2 -c -o src/interpreter.o src/interpreter.c
cc -std=c11 -Wall -Wextra -O2 -c -o src/main.o src/main.c
cc -std=c11 -Wall -Wextra -O2 -o minilang src/lexer.o src/ast.o src/parser.o src/interpreter.o src/main.o -lm
```

### 5.5 概念五：怎么证明"拆分前后行为一致"

这是本步最该学的一点：**重构必须有证据**。

`013/run.sh` 干三件事：

1. 用同一套 `tests/` 跑**单文件版**；
2. 用同一套 `tests/` 跑**多文件版**；
3. **对拍**：两个版本跑每个用例，逐字节比较 stdout、stderr 和退出码。

第 3 步的实现：

```sh
for f in tests/*.ml demo.ml; do
    s1=0; "./$BIN_SINGLE" "$f" "$mode" >"$TMP/a.out" 2>"$TMP/a.err" || s1=$?
    s2=0; "./$BIN_MULTI"  "$f" "$mode" >"$TMP/b.out" 2>"$TMP/b.err" || s2=$?

    if ! cmp -s "$TMP/a.out" "$TMP/b.out" || ! cmp -s "$TMP/a.err" "$TMP/b.err" || [ "$s1" != "$s2" ]; then
        echo "  ❌ $(basename "$f")：两个版本输出不一致"
    fi
done
```

注意用的是 `cmp -s`（逐字节比较）而不是 `diff`——
`diff` 是给人看的，`cmp` 是给机器用的。

真实输出：

```
▶ 对拍：单文件版 vs 多文件版
  ✅ 两个版本输出逐字节一致
```

**这为什么算证据？** 因为：

- 两个版本跑的是**同一批输入**；
- 比对的是**完整的输出**（stdout + stderr + 退出码），不是"看起来一样"；
- 用例覆盖了正常执行**和**各种错误路径（错误信息在 stderr 里，
  只要有一个字不同就会被 `cmp` 逮到）。

### 5.6 概念六：函数体一个字都没改

拆分是"搬家"，不是"重写"。对照两个版本可以验证：

| | 单文件版 | 多文件版 |
|---|---|---|
| `lex()` 的函数体 | 在 `minilang.c` | 在 `src/lexer.c` |
| 函数体内容 | — | **逐字相同** |
| 变化 | — | 去掉 `static`、加 `#include "minilang.h"` |

也就是说，本次重构的 diff 里**没有一行是逻辑改动**。
所有行为差异都只能来自"声明和定义对不上"这类链接错误——
而那些错误在编译期就会被抓住，**根本跑不起来**。

**这正是分模块的好处**：编译器会帮你检查所有跨模块的接口。
单文件时你少写一个 `static` 编译器不会吭声，多文件时头文件里的声明
和 `.c` 里的定义对不上，链接期就会报错。

## 6. 动手：把代码加进来

### 6.1 改动清单

本步**不新增功能**，只做重构：

| 动作 | 从哪来 | 到哪去 |
|---|---|---|
| 宏定义、枚举、结构体 | `minilang.c` | `src/minilang.h` |
| 词法分析、错误报告 | `minilang.c` | `src/lexer.c` |
| AST 节点、打印树 | `minilang.c` | `src/ast.c` |
| 语法分析器 | `minilang.c` | `src/parser.c` |
| 求值、执行、内置函数、函数表 | `minilang.c` | `src/interpreter.c` |
| 读文件、模式分发、main | `minilang.c` | `src/main.c` |
| — | 新增 | `Makefile` |

### 6.2 分段讲解

#### 6.2.1 建目录，先放头文件

```sh
mkdir -p src
```

`minilang.h` 里放三样东西：**宏定义、类型定义、函数声明**。

```c
#ifndef MINILANG_H
#define MINILANG_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <stdarg.h>

/* 全局上限 */
#define MAX_TOKENS     20000
#define MAX_NODES      20000
...

/* Token */
typedef enum { ... } TokType;
typedef struct { ... } Token;

/* AST */
typedef enum { ... } NodeType;
typedef struct Node Node;
struct Node { ... };

/* 运行时的类型 */
typedef struct { ... } Value;
typedef struct { ... } Flow;
typedef struct { ... } Var;
typedef struct Env { ... } Env;
typedef double (*BuiltinFn)(double *args, Token tok);
typedef struct { ... } Builtin;
typedef struct { ... } Func;

/* 各个模块对外提供的接口 */
extern Token toks[MAX_TOKENS];
extern int   ntok;
...
void  lex(void);
void  error_at(int line, int col, const char *fmt, ...);
Node *node(NodeType ty, Token t);
void  dump_ast(Node *n, const char *prefix);
Node *expression(void);
Node *parse_program(void);
Value eval(Node *n, Env *e);
Flow  exec(Node *n, Env *e);
void  register_functions(Node *root);
void  env_init(Env *e, Env *parent);
void  env_free(Env *e);
char *read_file(const char *path);
int   mode_tokens(const char *path);
int   mode_ast(const char *path);
int   mode_run(const char *path);

#endif /* MINILANG_H */
```

**代码要点**：

- `extern` 表示"这个变量定义在别的 .c 文件里，这里只是声明"。
- `struct Node` 在头文件里要**完整定义**（因为别的模块要 `sizeof` 它、
  要访问它的字段），而 `Env`、`Func` 也一样。
- 头文件里的 `#include` 只放**这个头文件自己需要**的类型
  （`stdio.h` 因为要用 `FILE` 和 `printf`）。

#### 6.2.2 切 `lexer.c`

把 `minilang.c` 里"Token + 错误处理 + 词法分析 + type_name"这一段搬过来，
去掉 `static`（该公开的），加上：

```c
#include "minilang.h"
```

**注意**：`print_source_line` 是**只有 lexer.c 自己用**的，
所以它**保持 `static`**：

```c
static void print_source_line(int line, int col) { ... }
```

**怎么判断该不该去掉 `static`？** 问自己一句：
"**别的 .c 文件要调用它吗？**"

| 函数 | 别的文件要用吗 | 处理 |
|---|---|---|
| `lex()` | 要（`main.c` 调用） | 去掉 `static` |
| `error_at()` / `fatal_at()` | 要（parser.c、interpreter.c 都调） | 去掉 `static` |
| `type_name()` | 要（ast.c、main.c 都调） | 去掉 `static` |
| `print_source_line()` | 不要 | **保持 `static`** |
| `is_id_start()` / `is_id_char()` | 不要 | 保持 `static` |
| `add_token()` | 不要 | 保持 `static` |

#### 6.2.3 切 `ast.c`

`node()` 和 `dump_ast()` 要被 parser.c 和 main.c 调用 → 去掉 `static`。
节点池 `nodes[]` / `nnodes` 只有 ast.c 用 → 保持 `static`。

```c
static Node nodes[MAX_NODES];
static int  nnodes;

Node *node(NodeType ty, Token t) { ... }     /* 公开 */
void  dump_ast(Node *n, const char *prefix) { ... }   /* 公开 */
```

#### 6.2.4 切 `parser.c`

`expression()` 和 `parse_program()` 要被 main.c 调用 → 去掉 `static`。
其余（`primary`、`unary`、`bin_level`、`mul`、`add`…）都是内部的 → 保持 `static`。

**注意**：`statement()` 和 `block()` 互相调用，而且 `statement()` 要调用
`expression()`——它们都在 parser.c 里，所以保持 `static` 没问题。

#### 6.2.5 切 `interpreter.c`

这个文件最大（417 行），因为它装着：作用域、变量、`Value`、`Flow`、
内置函数表、用户函数表、`eval`、`exec`。

要公开的：`eval`、`exec`、`register_functions`、`env_init`、`env_free`。

**为什么 `env_init` / `env_free` 要公开？** 因为 `main.c` 里的
`mode_run` 要创建全局环境：

```c
Env global;
env_init(&global, NULL);
...
env_free(&global);
```

保持 `static` 的：`env_push`、`find_var`、`get_var`、`define_var`、`set_var`、
`print_template`、`bi_*`、`find_builtin`、`find_func`、`number`、`noflow`。

**还有一处容易漏**：第 10 步为了调用 `exec` 加的前向声明

```c
static Flow exec(Node *n, Env *e);
```

在多文件版本里，这个前向声明变成了 `minilang.h` 里的正式声明：

```c
Flow exec(Node *n, Env *e);
```

所以 `interpreter.c` 里**不再需要**那句前向声明了——
这正是头文件的价值：**一份声明，所有模块共享**。

#### 6.2.6 切 `main.c`

`read_file`、`mode_tokens`、`mode_ast`、`mode_run` 是 `main()` 用的。
`main()` 本身在头文件里**不声明**（只有 main.c 自己用）。

`print_escaped()`（第 1 步加的、显示字符串转义的辅助函数）只有 main.c 用 → `static`。

#### 6.2.7 写 Makefile

见 5.4 的完整讲解。

**最容易踩的两个坑**：

1. **命令必须用 Tab 缩进**。用空格的话 make 会报
   "missing separator"（缺少分隔符）。
2. **`-lm` 要放在源文件后面**。

### 6.3 完整 diff

本步的单文件版和第 12 步**完全相同**（`diff` 为空），所以没有"从上一步改过来"的 diff。
真正的改动是**新增** `src/` 目录和 `Makefile`——也就是下面这些文件：

- `src/minilang.h` ← **最重要的一个**，下面贴全文
- `src/lexer.c`
- `src/ast.c`
- `src/parser.c`
- `src/interpreter.c`
- `src/main.c`
- `Makefile`

它们的内容就是从 `minilang.c` 里**原样搬过去**的——
唯一的系统性改动是：公开的加进头文件、私有的保持 `static`、
每个 `.c` 开头加 `#include "minilang.h"`。

先看最关键的 `src/minilang.h`（全新文件，161 行）：

<details>
<summary>点开看 <code>src/minilang.h</code> 全文</summary>

```diff
--- /dev/null
+++ b/src/minilang.h
@@ -0,0 +1,161 @@
+/*
+ * MiniLang —— 用 C 实现一门小型解释型语言
+ *
+ * 这是拆成多文件之后的公共头文件：
+ * 所有模块共用的常量、类型和函数声明都放在这里。
+ *
+ * 原来单文件里的 static 变量和函数，在这里变成"头文件里 extern 声明、
+ * 某一个 .c 文件里定义"；只在本文件里用的辅助函数仍然保持 static。
+ */
+#ifndef MINILANG_H
+#define MINILANG_H
+
+#include <stdio.h>
+#include <stdlib.h>
+#include <string.h>
+#include <ctype.h>
+#include <math.h>
+#include <stdarg.h>
+
+/* ------------------------------------------------------------------ */
+/* 全局上限                                                            */
+/* ------------------------------------------------------------------ */
+#define MAX_TOKENS     20000
+#define MAX_NODES      20000
+#define MAX_TEXT       4096
+#define MAX_PARAMS     32
+#define MAX_VARS       2048
+#define MAX_NAME       128
+#define MAX_FUNCS      256
+#define MAX_CALL_DEPTH 128
+#define MAX_LOOP_STEPS 1000000
+#define MAX_SHOW_COLS 160     /* 出错时一行最多显示多少列源码 */
+
+/* ------------------------------------------------------------------ */
+/* Token                                                               */
+/* ------------------------------------------------------------------ */
+typedef enum {
+    T_EOF, T_NUM, T_ID, T_STR,
+    T_LET, T_FN, T_RETURN, T_IF, T_ELSE, T_FOR,
+    T_LP, T_RP, T_LB, T_RB, T_COMMA, T_SEMI,
+    T_PLUS, T_MINUS, T_STAR, T_SLASH, T_PERCENT,
+    T_ASSIGN, T_EQ, T_NE, T_LT, T_LE, T_GT, T_GE,
+    T_AND, T_OR, T_NOT
+} TokType;
+
+typedef struct {
+    TokType type;
+    char    text[MAX_TEXT];
+    double  num;
+    int     line, col;
+} Token;
+
+/* ------------------------------------------------------------------ */
+/* AST                                                                 */
+/* ------------------------------------------------------------------ */
+typedef enum {
+    N_NUM, N_STR, N_VAR, N_UNARY, N_BINARY, N_CALL,
+    N_BLOCK, N_LET, N_ASSIGN, N_EXPR, N_IF, N_FOR, N_FN, N_RETURN
+} NodeType;
+
+typedef struct Node Node;
+struct Node {
+    NodeType type;
+    Token   tok;
+    double  num;
+    char    text[MAX_TEXT];
+    int     op;
+    Node   *a, *b, *c;
+    Node   *items[128];
+    int     count;
+    char   *params[MAX_PARAMS];
+    int     nparams;
+};
+
+/* ------------------------------------------------------------------ */
+/* 运行时的类型                                                        */
+/* ------------------------------------------------------------------ */
+typedef struct {
+    double value;
+    int    is_string;
+    char   string[MAX_TEXT];
+} Value;
+
+typedef struct {
+    int   returned;
+    Value value;
+} Flow;
+
+typedef struct {
+    char   name[MAX_NAME];
+    double value;
+} Var;
+
+typedef struct Env {
+    Var   *vars;
+    int    count;
+    int    cap;
+    struct Env *parent;
+} Env;
+
+typedef double (*BuiltinFn)(double *args, Token tok);
+
+typedef struct {
+    const char *name;
+    int         min_args;
+    int         max_args;
+    BuiltinFn   fn;
+} Builtin;
+
+typedef struct {
+    char  name[MAX_NAME];
+    Node *decl;
+} Func;
+
+/* ------------------------------------------------------------------ */
+/* lexer.c                                                             */
+/* ------------------------------------------------------------------ */
+extern Token toks[MAX_TOKENS];
+extern int   ntok;
+extern int   pos;
+extern const char *src;
+extern int   src_line, src_col;
+extern int   had_error;
+extern int   loop_steps;
+
+void        lex(void);
+void        error_at(int line, int col, const char *fmt, ...);
+void        fatal_at(Token t, const char *fmt, ...);
+const char *type_name(TokType t);
+char       *dupstr(const char *s);
+
+/* ------------------------------------------------------------------ */
+/* ast.c                                                               */
+/* ------------------------------------------------------------------ */
+Node *node(NodeType ty, Token t);
+void  dump_ast(Node *n, const char *prefix);
+
+/* ------------------------------------------------------------------ */
+/* parser.c                                                            */
+/* ------------------------------------------------------------------ */
+Node *expression(void);
+Node *parse_program(void);
+
+/* ------------------------------------------------------------------ */
+/* interpreter.c                                                       */
+/* ------------------------------------------------------------------ */
+Value eval(Node *n, Env *e);
+Flow  exec(Node *n, Env *e);
+void  register_functions(Node *root);
+void  env_init(Env *e, Env *parent);
+void  env_free(Env *e);
+
+/* ------------------------------------------------------------------ */
+/* main.c                                                              */
+/* ------------------------------------------------------------------ */
+char *read_file(const char *path);
+int   mode_tokens(const char *path);
+int   mode_ast(const char *path);
+int   mode_run(const char *path);
+
+#endif /* MINILANG_H */
```

</details>

其余 5 个 `.c` 文件的完整内容在 `013/src/` 目录里，
它们和 `minilang.c` 对应段落的差异只有两条：
**去掉了该公开的 `static`**、**开头加了 `#include "minilang.h"`**。

<details>
<summary>点开看单文件版和多文件版的逐字节对拍结果</summary>

```
▶ 对拍：单文件版 vs 多文件版
  ✅ 两个版本输出逐字节一致
```

</details>

### 6.4 插入位置说明

本步是"新建文件"，没有插入位置的问题。目录结构：

```
013/
├── minilang.c        # 保留：单文件版（= 第 12 步）
├── Makefile          # 新增
└── src/              # 新增
    ├── minilang.h
    ├── lexer.c
    ├── ast.c
    ├── parser.c
    ├── interpreter.c
    └── main.c
```

## 7. 测试

`./run.sh` 的真实输出：

```
▶ 编译单文件版
▶ 编译多文件版（make）
▶ 用例（单文件版，模式：run）
  ✅ case_01_syntax_error.ml      [run]
  ✅ case_02_undefined_var.ml     [run]
  ✅ case_03_type_error.ml        [run]
  ✅ case_04_divide_by_zero.ml    [run]
  ✅ case_05_argument_error.ml    [run]
  ✅ case_06_missing_semicolon.ml [run]
▶ 用例（多文件版，模式：run）
  ✅ case_01_syntax_error.ml      [run]
  ✅ case_02_undefined_var.ml     [run]
  ✅ case_03_type_error.ml        [run]
  ✅ case_04_divide_by_zero.ml    [run]
  ✅ case_05_argument_error.ml    [run]
  ✅ case_06_missing_semicolon.ml [run]
▶ 对拍：单文件版 vs 多文件版
  ✅ 两个版本输出逐字节一致
▶ demo.ml
  | sum = 15
  | factorial = 120
  | sqrt = 9

✅ 全部通过（14 项）
```

`tests/` 目录是**从第 12 步原样复制**过来的，一个字节都没改——
这也顺便证明了"第 12 步的行为没有因为拆分而改变"。

## 8. 易错点与坑

### 坑 1：Makefile 里用空格缩进

**现象**：

```
Makefile:7: *** missing separator.  Stop.
```

**原因**：Makefile 的命令行**必须**用 Tab 缩进。编辑器可能自动把 Tab 变成 8 个空格。

**正确写法**：在编辑器里把缩进设成"只插入 Tab"，或者写 Makefile 时手动按 Tab。

### 坑 2：`-lm` 放在源文件前面

**现象**：

```
Undefined symbols for architecture x86_64:
  "_sqrt", referenced from:
     _bi_sqrt in interpreter.o
```

**原因**：链接器**从左到右**扫描，遇到未解析的符号时去**后面**的库里找。
`-lm` 在前面，`sqrt` 就还没被解析出来，链接器不会回头找。

**正确写法**：`cc ... -o minilang src/*.o -lm`。

### 坑 3：该公开的函数忘了去掉 `static`

**现象**：`main.c` 里调用 `lex()` 报 "undefined reference to `lex'"。

**原因**：`lex()` 在 lexer.c 里还是 `static`，链接器根本看不到它。

**正确写法**：判断标准是"**别的 .c 文件要不要用**"，
要就在头文件里声明并去掉 `static`。

### 坑 4：不该公开的函数去掉了 `static`

**现象**：能编译，但符号表里多了一堆本该私有的符号；
如果两个模块定义了同名函数，链接期会报 "duplicate symbol"。

**原因**：图省事把 `static` 全删了。

**正确写法**：只有**确实跨模块使用**的才公开。
`find_var`、`env_push` 这些内部辅助函数保持 `static`。

### 坑 5：忘了在头文件里 `#include` 需要的标准库

**现象**：`minilang.h` 里用了 `FILE` 但没 `#include <stdio.h>`，
别的 .c 文件 include 它时报 "`FILE` undeclared"。

**原因**：头文件用了什么类型，就得自己 include 对应的头文件——
**不能指望包含者已经 include 过了**。

**正确写法**：`minilang.h` 里 include 它自己用到的所有标准库。

### 坑 6：改了头文件但没重新编译

**现象**：改了 `minilang.h` 里的结构体定义，
但 `make` 只重编了部分文件，链接时结构体大小对不上，行为诡异。

**原因**：Makefile 里没写头文件依赖。

**正确写法**：

```make
$(SRCDIR)/%.o: $(SRCDIR)/%.c $(DEPS)
	$(CC) $(CFLAGS) -c -o $@ $<
```

`$(DEPS)` 就是 `$(SRCDIR)/minilang.h`。

### 坑 7：拆分时手抖改了一行代码

**现象**：两个版本行为不一致。

**原因**：搬家的时候手滑改了某个逻辑。

**正确写法**：**用对拍来发现它**——这就是 `run.sh` 存在的意义。
纯靠眼睛检查 1281 行的搬家是不可能靠谱的。

## 9. 自测题

### 9.1 概念题

**Q1.** 拆分后，`static` 的含义变了吗？

<details><summary>参考答案</summary>

`static` 的**语言含义没变**（还是"这个翻译单元内部可见"），
但**效果变了**：单文件时翻译单元是整个程序，拆开后翻译单元是每个 `.c` 文件。
所以原来"全程序私有"的 `static` 变量，拆开后要么去掉 `static` 变成公开
（并写进头文件），要么在不需要跨模块时继续保持 `static`。

</details>

**Q2.** 为什么 `print_source_line` 保持 `static`，而 `error_at` 要公开？

<details><summary>参考答案</summary>

判断标准是"**别的 .c 文件要不要调用它**"。
`print_source_line` 只有 lexer.c 自己在用（error_at 和 fatal_at 内部调用），
所以保持 `static`；`error_at` / `fatal_at` 被 parser.c、interpreter.c 调用，
所以必须公开。

</details>

**Q3.** 头号防重（`#ifndef`）防的是什么？

<details><summary>参考答案</summary>

防的是同一个头文件被**重复处理**。
头文件之间互相 include 时（或者多个 .c 间接 include 同一个头文件），
没有防重就会导致类型重复定义、编译报错。
`#ifndef MINILANG_H` 保证内容只被处理一次。

</details>

### 9.2 预测题（改一行会发生什么）

**Q4.** 把 Makefile 里的 `-lm` 挪到源文件前面，编译会怎样？

<details><summary>参考答案</summary>

链接失败：
```
Undefined symbols for architecture x86_64:
  "_sqrt", referenced from: _bi_sqrt in interpreter.o
  "_floor", ...
```
链接器从左到右扫描，`-lm` 在前面时 `sqrt`/`floor` 这些符号还没被解析出来。

</details>

**Q5.** 把 `interpreter.c` 里的 `static void env_push(...)` 的 `static` 去掉，
但**不**在 `minilang.h` 里加声明，编译会怎样？

<details><summary>参考答案</summary>

能编译通过，行为也不变——因为没有别的模块调用 `env_push`，
它只是一个"公开但没人用"的符号。
但这是**坏味道**：应该保持 `static`。
真正的错误是反过来：去掉 `static` 却在别的 .c 里调用它，
却忘了在头文件里声明——那会报 "implicit declaration" 警告。

</details>

**Q6.** 把 `Makefile` 里的 `$(DEPS)` 从依赖列表里删掉，改了 `minilang.h` 之后 `make` 会怎样？

<details><summary>参考答案</summary>

`make` 只会在 `.c` 文件比 `.o` 新时重新编译。
改了 `minilang.h`（时间戳更新）但 `.c` 没变，`.o` 就不会被重建——
用的是**旧的**结构体定义。
如果新头文件改了结构体大小，链接期可能不报错，但运行行为诡异。
这就是"改了头文件必须重编所有依赖它的 .o"的原因。

</details>

### 9.3 动手题

**Q7.** 把 `src/lexer.c` 里的 `error_at` 改成 `static`（即从公开变私有），
然后 `make`，会看到什么？

<details><summary>参考答案</summary>

编译能过（lexer.c 自己也用 `error_at`），但**链接失败**：
```
Undefined symbols for architecture x86_64:
  "_error_at", referenced from:
     _expect in parser.o
```
因为 parser.c 里的 `expect()` 调用 `error_at`，而 `error_at` 被藏在 lexer.c 里了。

</details>

**Q8.** 给 `src/` 加一个新的模块：把 `interpreter.c` 里的**内置函数表**
（`Builtin` 结构、`builtins[]`、`bi_*`、`find_builtin`）单独拆到 `src/builtins.c`。
然后让 `run.sh` 的对拍依然通过。

<details><summary>参考答案</summary>

要做三件事：
1. 在 `minilang.h` 里把 `Builtin`、`BuiltinFn` 的定义和 `find_builtin` 的声明挪过去；
2. 新建 `src/builtins.c`，`#include "minilang.h"`，把那几十行搬过去；
3. 在 `Makefile` 的 `SRCS` 里加上 `$(SRCDIR)/builtins.c`。

`find_builtin` 要被 `interpreter.c` 调用 → 公开；
`bi_*` 只有表里用 → 保持 `static`；
`builtins[]` 和 `nbuiltins` 只有本文件用 → 保持 `static`。

</details>

**Q9.** 在 `013/` 下用 `make test` 跑一遍测试（`test:` 目标已经写好了）。

预期：`run.sh` 的全部输出，和直接跑 `./run.sh` 一致。

<details><summary>参考答案</summary>

一致。`make test` 就是先 `make` 再 `./run.sh`。

</details>

### 9.4 延伸题

**Q10.** 如果要给 MiniLang 加一个"模块系统"（`import foo;`），
你会怎么利用这套多文件结构？

<details><summary>参考答案</summary>

可以给 `Func` 表加一个"模块"字段，或者干脆让每个模块是一组带前缀的函数名
（`math.sqrt` 这种）。实现上需要：
1. 词法分析器支持 `.` 运算符；
2. 语法分析器支持带点的调用；
3. `find_func` 支持按"模块名.函数名"查找。

现有的模块化结构（头文件声明接口、每个 .c 一个职责）会让这些改动
分散到明确的几个文件里，而不是全塞进一个 1281 行的文件——
这就是本步的意义。

</details>

## 10. 本步的设计取舍

- **保留单文件版**：很多场景下单文件更方便（复制粘贴就能用）。
  两个版本并存，用测试证明它们等价，是很实用的做法。
- **`make` 而不是手敲编译命令**：6 个 `.o` 文件手敲一遍很痛苦，
  而且改了头文件还得手动重编所有依赖。
- **不引入构建系统**：CMake、Meson、Bazel 都很强大，但对这个规模的教学项目太重了。
  一个 38 行的 Makefile 足够。
- **不做自动生成头文件**：真实项目里会用 `csmith`、`bear` 之类的工具
  根据 include 关系自动维护头文件依赖。教学版手写 `$(DEPS)` 就好。

## 11. 收官：接下来可以往哪里走

你已经写完了一门语言。剩下的路很长，下面是几个方向，按难度排序：

### 容易上手的

1. **字符串作为一等类型**：现在字符串只能出现在 `print` 里。
   给 `Value` 加一个真正的字符串支持，让 `let s = "hi";` 合法，
   然后支持 `s + "x"`、`len(s)` 之类的操作。
2. **数组**：`let a = [1, 2, 3];`，加上下标 `a[0]` 和 `a[i] = 5`。
   这需要在词法分析器加 `[` `]`，语法分析器加下标表达式。
3. **`break` / `continue`**：给 `Flow` 加两个标志位（见第 11 步的延伸题）。
4. **`++` / `--`**：现在 `for` 的更新部分只能写 `i = i + 1`。
   加一对 token 和两个 AST 节点就行。
5. **REPL**：读一行执行一行，不用每次重新 `read_file`。

### 中等难度

6. **错误恢复**：一次报多个错误（见第 12 步的延伸题）。
   这是往"像样的编译器"迈的第一步。
7. **更合理的内存管理**：现在 `dupstr` 的参数名泄漏，
   `Token`/`Node` 各占几十 MB 静态内存。可以换成 arena 分配。
8. **闭包**：现在函数能读外层变量，但只是"读"当时的状态。
   真正的闭包要把环境一起捕获。
9. **模块和包**：`import` 别的 `.ml` 文件。

### 有点挑战

10. **字节码虚拟机**：把 AST 编译成栈式字节码再执行。
    好处是能加个"JIT 编译器"，坏处是多了一层要设计的东西。
    Lua、Python、JS 都走过这条路。
11. **类型系统**：给语言加上 `int`/`float`/`string`/`bool` 的区分，
    写一个独立的语义分析阶段（README 第一节就提到过这个理想）。
12. **并行执行**：让 `for` 循环的各次迭代并行。
    （顺便说，动态数组的 `Env` 是**共享**的，并行时得加锁——
    这就是为什么很多语言干脆用不可变的持久化数据结构。）

### 最后

不管往哪走，**先写测试**。

这套教程里每一个步骤都有 `tests/` 目录，每一步做完 `./run.sh` 一跑就知道
自己有没有搞坏什么。没有测试的重构就是闭眼开车——
第 13 步那个"逐字节对拍"就是最好的例子。

---

## 附：13 步的代码演进一览

| 步 | 新增了什么 | 代码行数变化 |
|---|---|---|
| 001 | 词法分析器 | ~200 |
| 002 | AST + 表达式语法分析 | ~330 |
| 003 | 求值器 | ~400 |
| 004 | 比较与逻辑运算 | ~450 |
| 005 | 语句、作用域、运行时骨架 | ~640 |
| 006 | if / else | ~690 |
| 007 | for 循环 | ~750 |
| 008 | 内置函数表 | ~840 |
| 009 | 字符串与 print 模板 | ~900 |
| 010 | 用户函数 | ~1010 |
| 011 | return 与递归 | ~1050 |
| 012 | 错误诊断 | ~1100 |
| 013 | 模块化重构 | 1281（单文件）/ 1332（6 文件 + 头文件） |

**回头看这 13 步**，最值得记住的不是任何一个语法，
而是那两个"提前定型"的决定：

- 第 3 步就把 `Value` 的字符串字段预留好了，所以第 9 步加字符串时
  **一个函数签名都不用改**；
- 第 5 步就把 `Flow` 定型好了，所以第 11 步加 `return` 时
  **`exec()` 的签名和 7 个调用点一个都不用改**。

**为将来买单**，是这套教程里最值钱的一课。
