# 第 1 步 · 词法分析器

> 进度 █░░░░░░░░░ 1/13 ｜ 上一节：[MiniLang 项目总览](../README.md) ｜ 下一节：[第 2 步 · 表达式语法分析 → AST](../002/README.md)

## 1. 本步目标

读完你将能够：

- 让 `./minilang demo.ml tokens` 把一份 MiniLang 源码打印成一串 **token（词法单元）**，
  每一行都带上 `[行:列]` 位置。
- 正确区分**关键字**（`let` / `fn` / `return` / `if` / `else` / `for`）和**标识符**，
  并知道 `let` 在你的语言里是保留字，不能拿来当变量名。
- 正确识别**双字符运算符**（`==` `!=` `<=` `>=` `&&` `||`），不会被 `=` 和 `==` 混淆。
- 跳过 `//` 行注释，正确处理字符串字面量里的转义序列（`\n` `\t` `\"` `\\`）。
- 对一个**非法字符**给出带行列号的错误信息，并以退出码 `1` 结束。
- 理解本步定下的三件"地基"（错误函数、模式表、词法器一次做全），
  它们决定了后面 12 步的代码长什么样。

## 2. 运行方式

```sh
cd 001
./run.sh                 # 编译 + 跑全部用例 + 跑 demo
./minilang demo.ml      # 默认 run 模式（本步还没实现，会提示）
./minilang demo.ml tokens
```

> `run.sh` 会把编译产物放在 `.minilang`（前面有个点，是隐藏文件），
> 因为根目录下已经有一个叫 `minilang` 的可执行文件了。第 13 步引入 `Makefile`
> 之后，产物名会变回 `minilang`。

## 3. 现在的程序能做什么

| 模式 | 输入 | 输出 | 退出码 |
|---|---|---|---|
| `tokens` | 一个 `.ml` 文件 | 每行一个 token：`[行:列] 类型 原文`，数字额外附 `(数值)` | 0（成功）/ 1（有错） |
| `ast` | 一个 `.ml` 文件 | stderr 一行提示"语法分析尚未实现" | 1 |
| `run`（默认） | 一个 `.ml` 文件 | stderr 一行提示"解释执行尚未实现" | 1 |
| 参数个数不对 | — | stderr 打印用法 | 2 |
| 模式名不认识 | — | stderr 打印可用模式 | 2 |

真实运行（`./minilang demo.ml tokens`，`demo.ml` 内容见 `001/demo.ml`）：

```text
[1:1] LET      let
[1:5] ID       x
[1:7] ASSIGN   = 
[1:9] NUM      10  (10)
[1:11] SEMI     ;
...
[9:1] EOF      <文件结束>
```

（这里为了篇幅省略了中间几行，完整输出见第 7 节。）

注意 `=` 那一行末尾有一个空格：`printf("%-8s ", type_name(...))` 会把类型名补齐到 8 个字符，
然后再打一个空格，接着打印 token 的原始文本 `=`。所以你看到的确实是 `ASSIGN   = `（末尾带空格）。

## 4. 前置知识回顾

上一步之前我们什么都没写，这是这个教程的起点。你需要会的 C 知识有：

- `enum`（枚举）、`struct`（结构体）、`typedef`。
- `char *` 指针、`malloc` / `free`、字符串以 `'\0'` 结尾。
- `printf` / `fprintf` 的格式化输出。
- 文件读写：`fopen` / `fread` / `fseek` / `ftell`。

你**不需要**会编译原理——本步会一步步把概念铺开。

## 5. 本步引入的核心概念

### 5.1 为什么需要词法分析器？

编译器处理源码分三段，MiniLang 也一样：

```text
   源文本 (char *)
        │  ① 词法分析 (lexer)：把字符流切成 token 流
        ▼
   token 流 (Token[])
        │  ② 语法分析 (parser)：把 token 流变成语法树
        ▼
   语法树 (AST)
        │  ③ 语义分析 + 求值 (interpreter)：算出一个结果
        ▼
     结果 (Value)
```

词法分析器（lexer）只关心**字符**，不关心**含义**。它看到 `let x = 10;`，
只会告诉你："第 1 行第 1 列是个关键字 `let`，第 1 行第 5 列是个标识符 `x`……"，
它不知道 `x` 是什么，也不管 `let` 和 `=` 能不能配对成一条合法的语句。

**为什么不能跳过词法分析直接"按行切分字符串"？**
因为那样你就得在每个地方重新处理空格、注释、字符串、运算符的区别。
把"字符 → token"这件事一次做对，后面所有步骤都只面对结构化的 token，代码量会少一个数量级。

### 5.2 Token 结构：为什么必须带 line 和 col？

```c
typedef struct {
    TokType type;      /* 这个 token 是什么类别            */
    char    text[MAX_TEXT]; /* 原始文本（标识符/数字/字符串内容） */
    double  num;       /* 数字字面量的数值                 */
    int     line, col; /* 出错时要告诉用户"哪一行哪一列"    */
} Token;
```

四个字段各司其职：

| 字段 | 类型 | 谁在用 | 第几步真正用上 |
|---|---|---|---|
| `type` | `TokType` | 语法分析器判断"这是不是一个加号" | 第 2 步 |
| `text` | `char[MAX_TEXT]` | 打印 token、存标识符的名字 | 第 2 步 |
| `num` | `double` | 数字字面量的数值（`3.14` → `3.14`） | 第 3 步 |
| `line, col` | `int` | **错误定位** | 第 2 步开始报错就用 |

`line` / `col` 是**前瞻性设计（lookahead design）**的典型例子：
第 1 步的词法分析器一个错误都还没报过，但这两个字段已经在为"第 2 步的语法错误"
和"第 12 步的精确错误定位"做准备了。如果你第 1 步省掉它们，第 12 步想加
"出错那一行源码 + 一个 `^` 指示符"时，就得**回头重写整个词法分析器**——
那是几十处改动，而不是加一个字段。

这就是本教程反复强调的一条经验：**接口一旦定型，后面尽量只加不改。**

### 5.3 扫描主循环：游标是怎么推进的？

词法分析器的核心是一个大 `while` 循环，游标 `i` 从头扫到尾：

```text
源码:   l e t   x   =   1 0 ;
        ^                       ^
        i=0                  i=14 (i == '\0')

一次循环做三件事:
  ① 看当前字符 src[i]，判断它属于哪一类
  ② 处理这一类（可能吃掉 1 个或多个字符）
  ③ 推进 i，并同步更新 src_line / src_col
```

关键在于 `src_line` / `src_col` 必须**跟着 `i` 一起走**。
`src_line` 只在遇到 `'\n'` 时加一，`src_col` 在遇到 `'\n'` 时归 1，
其他情况（空格、制表符、普通字符）都只加 1：

```c
if (c == ' ' || c == '\t' || c == '\r') { i++; src_col++; continue; }
if (c == '\n') { i++; src_line++; src_col = 1; continue; }
```

注意 `\t`（制表符）只让列号加 **1**，虽然它在终端里可能显示成好几列宽。
这是一个刻意的简化：列号是"字符列"而不是"显示列"，好处是简单、不会因为
编辑器里 tab 宽度不同而对不上。代价是错误信息里的列号和编辑器显示的列可能有出入。

**每识别出一个 token 前，先把当前位置拍个快照：**

```c
int line = src_line, col = src_col;   /* 快照：这个 token 从哪开始 */
```

之后处理这个 token 的代码不管怎么推进 `i`，`line` / `col` 都还指着它的**起始位置**。
这是"起点定位"的标准做法——错误信息应该说"这个 `+` 在第几行第几列"，
而不是"处理完 `+` 之后我在第几行第几列"。

### 5.4 双字符运算符：为什么 `==` 必须先于 `=`

这是一步教程里最经典的一个坑。看这段代码：

```c
/* 运算符：先尝试双字符的（== != <= >= && ||），再退回单字符 */
TokType ty = T_EOF;
char two[3] = {0, 0, 0};
two[0] = (char)c; two[1] = src[i+1];
if      (!strncmp(two, "==", 2)) ty = T_EQ;
else if (!strncmp(two, "!=", 2)) ty = T_NE;
...
if (ty != T_EOF) { i += 2; src_col += 2; add_token(ty, two, 0, line, col); continue; }

switch (c) {
    ...
    case '=':  ty = T_ASSIGN;  break;
    ...
}
```

顺序不能反。假设你先试单字符 `=`，看到 `a == b` 时你会：

1. 匹配到 `=` → 产出 `T_ASSIGN`；
2. `i` 只前进 1；
3. 下一个字符还是 `=`，再产出一个 `T_ASSIGN`。

结果 `a == b` 变成了三个 token：`a` `=` `=` `b`——四个 token，而且语法上完全说不通。

**最长匹配（longest match / maximal munch）原则**：token 匹配总是取能匹配的最长的那一个。
所以双字符运算符一定要在单字符之前检查。

这里用的是 `strncmp(two, "==", 2)` 而不是 `strcmp`。
为什么？因为 `two` 只有 2 个字符有效（`two[2]` 是 `'\0'`），但 `strcmp` 会一直比到 `'\0'`，
看起来也能工作；不过 `strncmp` 更直白地表达了"我只比较前 2 个字符"的意图，
而且即使 `two` 没有以 `'\0'` 结尾也不会越界读。**在长度受限的缓冲区上，用 `strncmp` 更安全。**

一个细节：单字符运算符走 `switch` 之后，调用的是 `add_token(ty, two, 0, line, col)`——
传的是 `two` 而不是 `&c`。为什么？`two[0] == c`，所以 `two` 里正好也是这个字符，
而 `two` 一定是以 `'\0'` 结尾的（初始化为 `{0,0,0}`），可以直接当 C 字符串用。
如果传 `&c`（`char *`），`add_token` 里的 `snprintf` 只会复制 1 个字符，后面的垃圾就没了。

### 5.5 关键字：为什么先当标识符，再查表

```c
/* 标识符 / 关键字 */
if (is_id_start(c)) {
    size_t start = i;
    while (is_id_char((unsigned char)src[i])) { i++; src_col++; }
    char b[128];
    size_t n = i - start;
    if (n >= sizeof(b)) fatal_at((Token){.line=line, .col=col}, "标识符太长");
    memcpy(b, src + start, n); b[n] = 0;
    TokType ty = T_ID;                          /* 先假定是普通标识符 */
    if      (!strcmp(b, "let"))    ty = T_LET;  /* 再查表，是关键字就改类型 */
    else if (!strcmp(b, "fn"))     ty = T_FN;
    else if (!strcmp(b, "return")) ty = T_RETURN;
    else if (!strcmp(b, "if"))     ty = T_IF;
    else if (!strcmp(b, "else"))   ty = T_ELSE;
    else if (!strcmp(b, "for"))    ty = T_FOR;
    add_token(ty, b, 0, line, col);
    continue;
}
```

**为什么不用单独的 `is_keyword()` 函数先判断？** 因为关键字的词法形态和标识符**完全一样**：
都以字母开头、由字母数字下划线组成。如果先写一个 `is_keyword()` 去逐个比对，
你得先把整个词读出来才能判断——那还不如读完再查表。

所以流程是：**先用"标识符"的规则把整个词扫出来 → 拿到字符串 → 查关键字表 → 是关键字就换个 token 类型**。
这跟真实编译器里的做法一致（关键字就是"保留的标识符"）。

代价是关键字查找是**线性扫描**（`strcmp` 一个一个比）。对教学项目这完全够用；
真要优化，可以先按第一个字符分桶，或者用完美哈希，但那是过度设计。

### 5.6 `strtod` 为什么要检查 `end` 指针

```c
char *end = NULL;
double v = strtod(b, &end);
if (!end || *end) fatal_at((Token){.line=line, .col=col}, "非法数字 '%s'", b);
```

`strtod` 的第三个参数是 `char **endptr`：函数会把**成功解析到的部分结束位置**写进去。
惯用法的判断方式是 `*end == '\0'`，意思是"整个字符串都被吃掉了"。

为什么要检查？因为 `strtod` **遇到不能解析的字符就停下来，不会报错**。
比如传进去 `"12abc"`，`strtod` 会返回 `12.0`，`end` 指向 `'a'`，
剩下的 `"abc"` 被丢在一边——**静默地丢数据**。

`if (!end || *end)` 这个写法：
- `!end`：万一有人改了 `strtod` 的调用方式，`end` 可能是 `NULL`，解引用会崩；
  顺手做一次空指针保护。
- `*end`：非空说明**结尾处还有没被解析的字符**，就是非法数字。

不过在本步的代码里，数字的字符扫描已经保证了只可能是 `[0-9.]`，
所以这个检查是**防御性编程**——它保护的是"以后有人修改了扫描规则"这种情况。
真实编译器里这是必须的：Python 的 `1_000`、Rust 的 `1e5`、JS 的 `0x1F` 都会让这个检查生效。

### 5.7 `isdigit` 为什么要 `(unsigned char)` 强转

```c
static int is_id_start(int c) { return isalpha((unsigned char)c) || c == '_'; }
static int is_id_char(int c)  { return isalnum((unsigned char)c) || c == '_'; }
...
if (isdigit(c) || (c == '.' && isdigit((unsigned char)src[i+1]))) {
```

`isdigit` / `isalpha` / `isalnum` 来自 `<ctype.h>`，它们的参数类型是 `int`，
但**语义上要求传进去的是一个 `unsigned char` 代表的字符值，或者 `EOF`**。

C 标准说得很明白：如果传给 `isdigit` 的值既不是 `unsigned char` 范围的字符
（0~255），也不是 `EOF`，行为是**未定义（undefined behavior）**。

那为什么危险？看 `char` 的定义：

```c
char c = (unsigned char)src[i];   /* 本步的写法：安全 */
```

本步先把 `src[i]` 转成 `unsigned char`，所以 `c` 落在 0~255，永远合法。

但如果直接写 `char c = src[i];`，在 x86 上 `char` 默认是有符号的，
于是 UTF-8 里的中文第一个字节（`0xE4` = 228）会变成 `-28`。
把 `-28` 传给 `isdigit` 就**踩到了未定义行为**——实际在 glibc 上通常"碰巧"没事，
但编译器有权做任何事（用 `-fsanitize=undefined` 一测就报）。

所以规矩是：**只要调用 `<ctype.h>` 里的函数，就必须先把参数转成 `unsigned char`。**

顺便一提，本步里 `isdigit((unsigned char)src[i])` 和 `isdigit(c)` 混用是刻意的：
`c` 已经是 `unsigned char` 转过来的（值域 0~255），再转一次是无害的；
而 `src[i+1]` 是原始 `char`，必须转。这个不一致看着别扭，但都是为了正确性。

### 5.8 `//` 注释怎么跳过

```c
/* 行注释：// 后面到行尾全部忽略 */
if (c == '/' && src[i+1] == '/') {
    while (src[i] && src[i] != '\n') { i++; src_col++; }
    continue;
}
```

三个要点：

1. **必须先判断 `//` 再判断 `/`。** `/` 本身是除号运算符（`T_SLASH`），
   而 `//` 是注释。如果先处理 `/`，那 `//` 会被拆成两个除号。
   这和 5.4 的"双字符先于单字符"是同一个原理。
2. **循环条件里有 `src[i]`。** 不能只写 `src[i] != '\n'`，
   否则文件末尾没有换行符时会一直读到 `'\0'` 之后，越界。
3. **注释里的列号照样要加。** 因为下一个 token 的列号要在正确的位置上。
   注释虽然被丢弃，但它"占"了源码里的位置。

顺便说明：本语言**只支持 `//` 行注释，不支持 `/* */` 块注释**。
真实的 MiniLang 也一样——教学项目不需要那么多花哨。

### 5.9 三行"模式表"：为什么第 1 步就要把 `ast` 和 `run` 放进去

```c
static int mode_tokens(const char *path) { ... }
static int mode_ast(const char *path) { ... }     /* 本步：还没实现 */
static int mode_run(const char *path) { ... }     /* 本步：还没实现 */

int main(int argc, char **argv) {
    ...
    const char *path = argv[1];
    const char *mode = (argc == 3) ? argv[2] : "run";   /* 默认 run */

    if (!strcmp(mode, "tokens")) return mode_tokens(path);
    if (!strcmp(mode, "ast"))    return mode_ast(path);
    if (!strcmp(mode, "run"))    return mode_run(path);

    fprintf(stderr, "未知模式 '%s'，可用模式：tokens / ast / run\n", mode);
    return 2;
}
```

这是一个刻意的"**先定接口，后填实现**"的设计。三种模式从第 1 步起就定型，
**后面 12 步只会往 `mode_ast` / `mode_run` 的函数体里加代码，`main` 本身一个字都不改。**

好处是什么？想象一下你第 5 步才想起要支持模式参数：
那时你要改 `main`、改所有模式分支、还要重新考虑参数解析。
而现在，第 1 步就把入口约定写死了：

- 不带模式参数 → 默认 `run`；
- 模式名不认识 → 打印可用模式，退出码 `2`；
- 三个模式各返回一个退出码（`0` 成功 / `1` 出错）。

这三个约定一旦定下，**后面每一步都只是在既有骨架里填肉**。

而且本步 `mode_ast` / `mode_run` 打印的"尚未实现"提示，本身就是很好的教学点：
读者会立刻意识到"这个程序现在只能切 token，不能跑程序"。
等你走到第 5 步，看到同样的提示变成真的执行结果时，那种"哦原来模式表是这么用的"的感觉就来了。

### 5.10 本步埋下的三块"地基"

这三条是整个教程 13 步的骨架约定（后面每篇 README 都会复述）：

| 编号 | 约定 | 本步怎么落地的 |
|---|---|---|
| **INV-1** | 词法分析器**一次识别最终语言的全部 token**，后面再没改过 | 第 1 步就把 `== != <= >= && \|\| !`、6 个关键字、字符串全做了。第 9 步加字符串模板、第 12 步加错误定位，词法器**一行都没动** |
| **INV-6** | 第 1 步定型 `main` 的模式表（tokens/ast/run），后面只往表里加分支，`main` 主体不动 | 见 5.9。第 2、3、5 步都是往 `mode_ast` / `mode_run` 的函数体里塞代码 |
| **INV-7** | 第 1 步定型错误函数名和签名：`error_at(line,col,fmt,...)` / `fatal_at(Token,fmt,...)` | 见下面的错误处理小节 |

#### 错误处理：两个函数，两种命运

```c
/* 错误处理：从第一步就定好名字和签名，后面不再改 */
static void error_at(int line, int col, const char *fmt, ...) {
    va_list ap;
    fprintf(stderr, "错误 [%d:%d]: ", line, col);
    va_start(ap, fmt); vfprintf(stderr, fmt, ap); va_end(ap);
    fputc('\n', stderr);
    had_error = 1;
}

/* 致命错误：打印后直接结束程序（教学版采取最简单的策略） */
static _Noreturn void fatal_at(Token t, const char *fmt, ...) {
    va_list ap;
    fprintf(stderr, "错误 [%d:%d]: ", t.line, t.col);
    va_start(ap, fmt); vfprintf(stderr, fmt, ap); va_end(ap);
    fputc('\n', stderr);
    had_error = 1;
    exit(1);
}
```

| | `error_at(line, col, fmt, ...)` | `fatal_at(Token t, fmt, ...)` |
|---|---|---|
| 出错后 | 记下 `had_error = 1`，**继续往下走** | 直接 `exit(1)`，程序结束 |
| 退出码 | 由调用者决定（词法分析结束后的 `return had_error ? 1 : 0`） | 恒为 1 |
| 第几步定型 | 第 1 步 | 第 1 步 |

`fatal_at` 收的是 **`Token` 结构体**而不是 `(line, col)` 两个整数，
因为语法分析器手里拿到的就是 token，直接整个传进去省得拆。
这种"参数类型跟着调用场景走"的设计能省掉大量重复代码。

`_Noreturn` 是 C11 的关键字，意思是"这个函数不会返回"。
加上它之后，编译器知道 `fatal_at(...)` 之后的代码不可达，
可以做更好的优化，也能**警告你**"这条分支后面缺少 return"。

> ⚠️ 变参函数（`...`）的规矩：`fmt` 后面**必须**至少有 `va_start` / `va_end` 配对，
> 且 `fmt` 里 `%d` / `%s` 的个数必须和实际传的参数一一对应，多了少了都是未定义行为。
> `vfprintf` 的第 4 个参数是 `va_list`，不是 `...` 展开后的东西。

#### 一个小坑：`fatal_at` 传的是"临时构造"的 Token

```c
fatal_at((Token){.line=line, .col=col}, "无法识别字符 '%c'", c);
```

这是 **C99 的复合字面量（compound literal）**：在运行期构造一个 `Token` 临时对象，
把它的 `line` / `col` 设好，其余字段（`text` / `num`）为零值，然后把它传给 `fatal_at`。
`fatal_at` 只用到 `t.line` / `t.col`，所以其余字段是什么无所谓。

好处是**一行搞定**，不用先声明一个局部变量。
`text` 字段是 4096 字节的数组，用 `.line=` / `.col=` 指定初始化器时其余字段自动清零
（C 语言保证指定初始化器会把没提到的成员置零），所以不会读到垃圾。

---

## 6. 动手：把代码加进来

### 6.1 改动清单

第 1 步是起点，没有"上一步的代码"可以改——你要**新建整个 `minilang.c`**。
下面这张表是给你自己用的施工清单，照着它把文件一段段写出来：

| 位置锚点 | 改了什么 | 大概多少行 | 为什么 |
|---|---|---|---|
| 文件头 | 注释块：编译命令、语法速览、三种模式说明 | 22 | 让读者一眼看到"整个教程会长什么样" |
| `#include` 区 | `stdio/stdlib/string/ctype/stdarg` | 5 | `ctype` 给 `isdigit`，`stdarg` 给变参函数 |
| `#define` 区 | `MAX_TOKENS 20000`、`MAX_TEXT 4096` | 2 | 两个静态数组的容量上限 |
| `TokType` 枚举 | 31 个 token 类型 | 4 | 词法分析器的"输出词汇表" |
| `Token` 结构体 | `type/text/num/line/col` | 7 | 见 5.2 |
| 全局变量 | `toks/ntok/src/src_line/src_col/had_error` | 6 | 词法分析器的状态 |
| 错误处理 | `error_at` / `fatal_at` | 20 | 见 5.10 |
| `is_id_start` / `is_id_char` | 标识符字符判定 | 3 | 见 5.7 |
| `add_token` | 往 token 数组里追加一个 token | 9 | 唯一的 token 出口，顺带做上限检查 |
| `lex()` | 扫描主循环 | 118 | 本步的主体，见 5.3–5.8 |
| `type_name()` | token 类型 → 英文名字 | 34 | 打印时用；第 2 步起 AST 打印也复用它 |
| `read_file()` | 整个文件读进一个 `char *` | 14 | 词法分析器需要一个连续的、以 `\0` 结尾的缓冲区 |
| `print_escaped()` | 打印字符串时把 `\n` 显示成 `\\n` | 11 | 否则 token 列表里的换行会撑乱排版 |
| 三个模式函数 | `mode_tokens` / `mode_ast` / `mode_run` | 26 | 见 5.9 |
| `main()` | 参数解析 + 模式分发 | 17 | 见 5.9 |

合计约 330 行。

### 6.2 分段讲解

#### 第 1 段：Token 类型表（读代码前先想清楚"要认哪些词"）

```c
typedef enum {
    T_EOF, T_NUM, T_ID, T_STR,
    T_LET, T_FN, T_RETURN, T_IF, T_ELSE, T_FOR,
    T_LP, T_RP, T_LB, T_RB, T_COMMA, T_SEMI,
    T_PLUS, T_MINUS, T_STAR, T_SLASH, T_PERCENT,
    T_ASSIGN, T_EQ, T_NE, T_LT, T_LE, T_GT, T_GE,
    T_AND, T_OR, T_NOT
} TokType;
```

这 31 个类型就是**最终 MiniLang 的全部词**。第 1 步把它们一次认全（INV-1），
后面 12 步的词法器再没改过。原因是：每加一个 token 类型都要动 `TokType` 枚举、
`type_name()` 函数、`lex()` 的匹配逻辑三处，代价高；而**提前多认几个用不上的 token 几乎不要钱**。

`T_EOF`（end of file）是虚拟 token，表示"源码读完了"。
它的存在让语法分析器不用单独处理"数组越界"的情况——
每次读 token 都能拿到一个合法的 `T_EOF`。

**如果你写错了会怎样？**

- 少写一个 `T_NOT`（比如 `!` ），编译会报 `use of undeclared identifier 'T_NOT'`。
- 把 `T_EQ` 和 `T_ASSIGN` 的顺序写反（`T_ASSIGN` 在前），
  编译**不会报错**，但 `==` 的匹配逻辑就全乱了——这类 bug 最难查，因为它是"静默的行为错误"。

#### 第 2 段：`add_token`——唯一的 token 出口

```c
static void add_token(TokType type, const char *text, double num, int line, int col) {
    if (ntok >= MAX_TOKENS) { error_at(line, col, "Token 数量超出上限"); exit(1); }
    Token *t = &toks[ntok++];
    t->type = type;
    t->num  = num;
    t->line = line;
    t->col  = col;
    snprintf(t->text, sizeof(t->text), "%s", text ? text : "");
}
```

四个要点：

1. **上限检查在写入之前。** `if (ntok >= MAX_TOKENS)` 而不是 `>`。
   因为 `ntok` 是"下一个要用的下标"，等于 `MAX_TOKENS` 时已经满了。
2. **`toks` 是静态数组，不是指针数组。** `Token *t = &toks[ntok++]` 指向数组内部，
   生命周期和程序一样长，不会失效——这比 `malloc` 每个 token 简单得多。
   `MAX_TOKENS` 个 `Token`，每个 `Token` 里有个 4096 字节的 `text`，
   静态占用约 `20000 * (4096 + 24) ≈ 82 MB`……**这是本步唯一"奢侈"的地方**。
   教学项目里 BSS（未初始化数据段）不占磁盘、可执行文件不膨胀，
   操作系统按需分页，所以完全能接受。真实项目会改成"token 里只存指针 + 源文件偏移"。
3. **`snprintf` 而不是 `strcpy`。** `snprintf(t->text, sizeof(t->text), "%s", ...)`
   保证最多写 `sizeof(t->text) - 1 = 4095` 个字符 + 一个 `'\0'`，
   **永远不会越界**。`strcpy` 遇到超长字符串会直接踩内存。
   `snprintf` 还有一个好处：**超长时静默截断而不是报错**——这也是后面第 2 步
   要单独用 `memcpy` + 长度检查来处理标识符名字的原因。
4. **`text ? text : ""`** 处理 `text` 为 `NULL` 的情况。虽然本步所有调用都传了非空指针，
   但这是一行成本极低的保险。

**如果你写错了会怎样？**

- 漏了 `snprintf` 的 `sizeof` 参数，写成 `snprintf(t->text, 4096, "%s", text)`：
  现在恰好没事，但以后有人把 `MAX_TEXT` 改成 1024，这里就溢出了。
- 用 `strcpy`：`text` 超过 4096 字节时堆溢出，可能表现为随机崩溃或安全漏洞。

#### 第 3 段：`lex()` 主循环

完整代码在 6.3，这里只讲结构。`lex()` 是一个大 `while (src[i])` 循环，
每一轮按下面的顺序尝试（**顺序即优先级**）：

```text
 ① 空白字符？        → 跳过（列号照样推进）
 ② '\n'？            → 换行（行号 +1，列号归 1）
 ③ "//" 注释？       → 跳到行尾
 ④ 记下 line/col 快照
 ⑤ 数字？            → T_NUM
 ⑥ 标识符/关键字？    → T_ID 或某个 T_XXX
 ⑦ 字符串？"        → T_STR
 ⑧ 双字符运算符？    → T_EQ/T_NE/... （必须早于 ⑨）
 ⑨ 单字符运算符？    → switch
 ⑩ 都不是            → fatal_at("无法识别字符")
```

**如果你写错了会怎样？**

- **把 ⑧ 放到 ⑨ 后面**：`==` 被拆成两个 `=`，见 5.4。
- **忘记在 ⑩ 里报错，直接 `i++` 跳过**：未知字符被静默吞掉，
  `let x = 1 @ 2;` 会"成功"跑完，但 `@` 消失得莫名其妙。
  对比一下本步的 `case_06_bad_char.ml`：正确实现是
  `错误 [1:11]: 无法识别字符 '@'`，退出码 1。
- **在 ④ 之前就 `i++`**：`line` / `col` 会指错位置，错误信息的行列号全乱。

#### 第 4 段：字符串字面量

```c
if (c == '"') {
    i++; src_col++;
    char b[MAX_TEXT];
    size_t n = 0;
    int closed = 0;
    while (src[i]) {
        if (src[i] == '"') { i++; src_col++; closed = 1; break; }
        if (src[i] == '\n')
            fatal_at((Token){.line=line, .col=col}, "字符串不能直接跨行，请使用 \\n");
        char ch = src[i++]; src_col++;
        if (ch == '\\' && src[i]) {         /* 转义序列 */
            char e = src[i++]; src_col++;
            if      (e == 'n')  ch = '\n';
            else if (e == 't')  ch = '\t';
            else if (e == 'r')  ch = '\r';
            else if (e == '"' || e == '\\') ch = e;
            else fatal_at((Token){.line=line, .col=col}, "不支持的字符串转义 \\%c", e);
        }
        if (n + 1 >= sizeof(b)) fatal_at((Token){.line=line, .col=col}, "字符串太长");
        b[n++] = ch;
    }
    if (!closed) fatal_at((Token){.line=line, .col=col}, "字符串缺少结束引号");
    b[n] = 0;
    add_token(T_STR, b, 0, line, col);
    continue;
}
```

四个设计决定：

1. **字符串不许跨行。** 遇到 `\n` 直接报错。这不是 C 语言的要求，
   是 MiniLang 的语言设计——为了让错误定位简单（一个字符串在一行内），
   也是为了让"字符串里的 `\n` 只能是转义出来的"这条规则明确。
2. **字符串里存的是"处理后的内容"**，不是原始文本。
   `"hello\n"` 存进 `token.text` 的是一个真正的换行符 `0x0A`。
   打印时用 `print_escaped()` 把它重新显示成 `\n`，
   所以你在 tokens 输出里看到的还是 `"hello\n"`。
3. **`closed` 标志。** 因为字符串可能**跨行**才触发错误，也可能正常结束。
   循环结束时如果 `closed == 0`，说明是**文件结束**时还没遇到闭合引号——
   报"字符串缺少结束引号"。
4. **`n + 1 >= sizeof(b)`。** 为什么要 `+1`？
   因为最后还要写 `b[n] = 0`。当 `n == sizeof(b) - 1 = 4095` 时，
   再写一个字符就占了 `b[4095]`，然后 `b[4096] = 0` 就越界了。
   所以条件必须是 `n + 1 >= sizeof(b)`，而不是 `n >= sizeof(b)`。

**如果你写错了会怎样？**

- 忘了 `closed` 检查：`let s = "abc` （文件里没有换行）会**静默成功**，
  token 列表里出现一个没有闭合的字符串。
- 转义处理写成 `if (ch == '\\')` 后**忘了 `i++`**：
  `"\n"` 里的 `n` 会被当成普通字符收进去，字符串内容变成 `\n` 两个字符（反斜杠 + n）。
- 越界检查写成 `n >= sizeof(b)`：字符串长度正好 4095 时越界 1 字节。

### 6.3 完整代码

第 1 步没有"上一步"，所以这里没有 diff 可贴。下面是本步的**完整 `minilang.c`**，
你可以直接照着敲（或者对照 `001/minilang.c`）。

```c
/*
 * MiniLang —— 用 C 实现一门小型解释型语言
 *
 * 编译: cc -std=c11 -Wall -Wextra -O2 minilang.c -lm -o minilang
 * 运行: ./minilang 脚本.ml [tokens|ast|run]
 *
 * 语法速览:
 *   let x = 10;              定义变量
 *   x = x + 1;               赋值
 *   print("x = {x}\n");      打印（模板里只支持 {变量名}）
 *   if (x > 10) { ... } else { ... }
 *   for (let i = 0; i < 5; i = i + 1) { ... }
 *   fn add(a, b) { return a + b; }
 *
 * 三种运行模式:
 *   tokens  只做词法分析，打印 token 列表
 *   ast     词法 + 语法分析，打印抽象语法树
 *   run     完整执行程序（默认）
 *
 * 说明：文件头注释在教程所有步骤中保持完全一致，这样相邻两步的 diff
 * 就只剩下真正的功能改动，读者不会被无关的噪音干扰。
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdarg.h>

#define MAX_TOKENS 20000
#define MAX_TEXT   4096

/* ------------------------------------------------------------------ */
/* Token：词法分析器产出的最小单位                                     */
/* ------------------------------------------------------------------ */
typedef enum {
    T_EOF, T_NUM, T_ID, T_STR,
    T_LET, T_FN, T_RETURN, T_IF, T_ELSE, T_FOR,
    T_LP, T_RP, T_LB, T_RB, T_COMMA, T_SEMI,
    T_PLUS, T_MINUS, T_STAR, T_SLASH, T_PERCENT,
    T_ASSIGN, T_EQ, T_NE, T_LT, T_LE, T_GT, T_GE,
    T_AND, T_OR, T_NOT
} TokType;

typedef struct {
    TokType type;      /* 这个 token 是什么类别            */
    char    text[MAX_TEXT]; /* 原始文本（标识符/数字/字符串内容） */
    double  num;       /* 数字字面量的数值                 */
    int     line, col; /* 出错时要告诉用户"哪一行哪一列"    */
} Token;

static Token toks[MAX_TOKENS];   /* token 数组 */
static int   ntok;               /* 已经识别了多少个 token */
static const char *src;          /* 整份源代码 */
static int   src_line, src_col;  /* 扫描器当前所在的位置 */
static int   had_error;          /* 出过错没有 */

/* ------------------------------------------------------------------ */
/* 错误处理：从第一步就定好名字和签名，后面不再改                       */
/* ------------------------------------------------------------------ */
static void error_at(int line, int col, const char *fmt, ...) {
    va_list ap;
    fprintf(stderr, "错误 [%d:%d]: ", line, col);
    va_start(ap, fmt); vfprintf(stderr, fmt, ap); va_end(ap);
    fputc('\n', stderr);
    had_error = 1;
}

/* 致命错误：打印后直接结束程序（教学版采取最简单的策略） */
static _Noreturn void fatal_at(Token t, const char *fmt, ...) {
    va_list ap;
    fprintf(stderr, "错误 [%d:%d]: ", t.line, t.col);
    va_start(ap, fmt); vfprintf(stderr, fmt, ap); va_end(ap);
    fputc('\n', stderr);
    had_error = 1;
    exit(1);
}

/* 标识符可以以字母或下划线开头 */
static int is_id_start(int c) { return isalpha((unsigned char)c) || c == '_'; }
/* 标识符的后续字符可以是字母、数字或下划线 */
static int is_id_char(int c)  { return isalnum((unsigned char)c) || c == '_'; }

/* ------------------------------------------------------------------ */
/* 词法分析器                                                         */
/* ------------------------------------------------------------------ */
static void add_token(TokType type, const char *text, double num, int line, int col) {
    if (ntok >= MAX_TOKENS) { error_at(line, col, "Token 数量超出上限"); exit(1); }
    Token *t = &toks[ntok++];
    t->type = type;
    t->num  = num;
    t->line = line;
    t->col  = col;
    snprintf(t->text, sizeof(t->text), "%s", text ? text : "");
}

/* 把源文件扫描成一串 token */
static void lex(void) {
    ntok = 0; src_line = 1; src_col = 1;
    size_t i = 0;
    while (src[i]) {
        int c = (unsigned char)src[i];

        /* 空白字符：跳过，但列号要跟着走 */
        if (c == ' ' || c == '\t' || c == '\r') { i++; src_col++; continue; }
        if (c == '\n') { i++; src_line++; src_col = 1; continue; }

        /* 行注释：// 后面到行尾全部忽略 */
        if (c == '/' && src[i+1] == '/') {
            while (src[i] && src[i] != '\n') { i++; src_col++; }
            continue;
        }

        int line = src_line, col = src_col;

        /* 数字：整数或小数 */
        if (isdigit(c) || (c == '.' && isdigit((unsigned char)src[i+1]))) {
            size_t start = i;
            int dots = 0;
            while (isdigit((unsigned char)src[i]) || src[i] == '.') {
                if (src[i] == '.' && ++dots > 1) break;   /* 最多一个小数点 */
                i++; src_col++;
            }
            char b[128];
            size_t n = i - start;
            if (n >= sizeof(b)) fatal_at((Token){.line=line, .col=col}, "数字字面量太长");
            memcpy(b, src + start, n); b[n] = 0;
            char *end = NULL;
            double v = strtod(b, &end);
            if (!end || *end) fatal_at((Token){.line=line, .col=col}, "非法数字 '%s'", b);
            add_token(T_NUM, b, v, line, col);
            continue;
        }

        /* 标识符 / 关键字 */
        if (is_id_start(c)) {
            size_t start = i;
            while (is_id_char((unsigned char)src[i])) { i++; src_col++; }
            char b[128];
            size_t n = i - start;
            if (n >= sizeof(b)) fatal_at((Token){.line=line, .col=col}, "标识符太长");
            memcpy(b, src + start, n); b[n] = 0;
            TokType ty = T_ID;
            if      (!strcmp(b, "let"))    ty = T_LET;
            else if (!strcmp(b, "fn"))     ty = T_FN;
            else if (!strcmp(b, "return")) ty = T_RETURN;
            else if (!strcmp(b, "if"))     ty = T_IF;
            else if (!strcmp(b, "else"))   ty = T_ELSE;
            else if (!strcmp(b, "for"))    ty = T_FOR;
            add_token(ty, b, 0, line, col);
            continue;
        }

        /* 字符串字面量 */
        if (c == '"') {
            i++; src_col++;
            char b[MAX_TEXT];
            size_t n = 0;
            int closed = 0;
            while (src[i]) {
                if (src[i] == '"') { i++; src_col++; closed = 1; break; }
                if (src[i] == '\n')
                    fatal_at((Token){.line=line, .col=col}, "字符串不能直接跨行，请使用 \\n");
                char ch = src[i++]; src_col++;
                if (ch == '\\' && src[i]) {         /* 转义序列 */
                    char e = src[i++]; src_col++;
                    if      (e == 'n')  ch = '\n';
                    else if (e == 't')  ch = '\t';
                    else if (e == 'r')  ch = '\r';
                    else if (e == '"' || e == '\\') ch = e;
                    else fatal_at((Token){.line=line, .col=col}, "不支持的字符串转义 \\%c", e);
                }
                if (n + 1 >= sizeof(b)) fatal_at((Token){.line=line, .col=col}, "字符串太长");
                b[n++] = ch;
            }
            if (!closed) fatal_at((Token){.line=line, .col=col}, "字符串缺少结束引号");
            b[n] = 0;
            add_token(T_STR, b, 0, line, col);
            continue;
        }

        /* 运算符：先尝试双字符的（== != <= >= && ||），再退回单字符 */
        TokType ty = T_EOF;
        char two[3] = {0, 0, 0};
        two[0] = (char)c; two[1] = src[i+1];
        if      (!strncmp(two, "==", 2)) ty = T_EQ;
        else if (!strncmp(two, "!=", 2)) ty = T_NE;
        else if (!strncmp(two, "<=", 2)) ty = T_LE;
        else if (!strncmp(two, ">=", 2)) ty = T_GE;
        else if (!strncmp(two, "&&", 2)) ty = T_AND;
        else if (!strncmp(two, "||", 2)) ty = T_OR;
        if (ty != T_EOF) { i += 2; src_col += 2; add_token(ty, two, 0, line, col); continue; }

        switch (c) {
            case '(':  ty = T_LP;      break;
            case ')':  ty = T_RP;      break;
            case '{':  ty = T_LB;      break;
            case '}':  ty = T_RB;      break;
            case ',':  ty = T_COMMA;   break;
            case ';':  ty = T_SEMI;    break;
            case '+':  ty = T_PLUS;    break;
            case '-':  ty = T_MINUS;   break;
            case '*':  ty = T_STAR;    break;
            case '/':  ty = T_SLASH;   break;
            case '%':  ty = T_PERCENT; break;
            case '=':  ty = T_ASSIGN;  break;
            case '<':  ty = T_LT;      break;
            case '>':  ty = T_GT;      break;
            case '!':  ty = T_NOT;    break;
            default:   fatal_at((Token){.line=line, .col=col}, "无法识别字符 '%c'", c);
        }
        i++; src_col++;
        add_token(ty, two, 0, line, col);
    }
    add_token(T_EOF, "<文件结束>", 0, src_line, src_col);
}

/* token 类型的中文/英文名字，打印时用 */
static const char *type_name(TokType t) {
    switch (t) {
        case T_EOF:     return "EOF";
        case T_NUM:     return "NUM";
        case T_ID:      return "ID";
        case T_STR:     return "STR";
        case T_LET:     return "LET";
        case T_FN:      return "FN";
        case T_RETURN:  return "RETURN";
        case T_IF:      return "IF";
        case T_ELSE:    return "ELSE";
        case T_FOR:     return "FOR";
        case T_LP:      return "LP";
        case T_RP:      return "RP";
        case T_LB:      return "LB";
        case T_RB:      return "RB";
        case T_COMMA:   return "COMMA";
        case T_SEMI:    return "SEMI";
        case T_PLUS:    return "PLUS";
        case T_MINUS:   return "MINUS";
        case T_STAR:    return "STAR";
        case T_SLASH:   return "SLASH";
        case T_PERCENT: return "PERCENT";
        case T_ASSIGN:  return "ASSIGN";
        case T_EQ:      return "EQ";
        case T_NE:      return "NE";
        case T_LT:      return "LT";
        case T_LE:      return "LE";
        case T_GT:      return "GT";
        case T_GE:      return "GE";
        case T_AND:     return "AND";
        case T_OR:      return "OR";
        case T_NOT:     return "NOT";
    }
    return "?";
}

/* ------------------------------------------------------------------ */
/* 读文件                                                              */
/* ------------------------------------------------------------------ */
static char *read_file(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) { perror(path); exit(1); }
    if (fseek(fp, 0, SEEK_END) != 0) { perror("fseek"); exit(1); }
    long len = ftell(fp);
    if (len < 0 || len > 4 * 1024 * 1024) { fprintf(stderr, "文件太大或无法读取\n"); exit(1); }
    rewind(fp);
    char *b = (char *)malloc((size_t)len + 1);
    if (!b) { perror("malloc"); exit(1); }
    size_t got = fread(b, 1, (size_t)len, fp);
    fclose(fp);
    b[got] = 0;
    return b;
}

/* 把字符串里的转义字符显示出来，方便看 token */
static void print_escaped(const char *s) {
    putchar('"');
    for (; *s; s++) {
        if      (*s == '\n') printf("\\n");
        else if (*s == '\t') printf("\\t");
        else if (*s == '\r') printf("\\r");
        else putchar(*s);
    }
    putchar('"');
}

/* ------------------------------------------------------------------ */
/* 三种模式：tokens / ast / run                                        */
/* 模式表在第一步就定型，后面每一步只往上加分支，不动 main 的骨架       */
/* ------------------------------------------------------------------ */
static int mode_tokens(const char *path) {
    src = read_file(path);
    lex();
    for (int i = 0; i < ntok; i++) {
        Token *t = &toks[i];
        printf("[%d:%d] %-8s ", t->line, t->col, type_name(t->type));
        if (t->type == T_STR) print_escaped(t->text);
        else printf("%s", t->text);
        if (t->type == T_NUM) printf("  (%g)", t->num);
        putchar('\n');
    }
    free((void *)src);
    return had_error ? 1 : 0;
}

static int mode_ast(const char *path) {
    (void)path;
    fprintf(stderr, "语法分析尚未实现，请先用 tokens 模式查看词法结果\n");
    return 1;
}

static int mode_run(const char *path) {
    (void)path;
    fprintf(stderr, "解释执行尚未实现，请先用 tokens 模式查看词法结果\n");
    return 1;
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "用法: %s 脚本.ml [tokens|ast|run]\n", argv[0]);
        fprintf(stderr, "示例: fn square(x) { return x * x; } print(\"{square(5)}\\n\");\n");
        return 2;
    }
    const char *path = argv[1];
    const char *mode = (argc == 3) ? argv[2] : "run";

    if (!strcmp(mode, "tokens")) return mode_tokens(path);
    if (!strcmp(mode, "ast"))    return mode_ast(path);
    if (!strcmp(mode, "run"))    return mode_run(path);

    fprintf(stderr, "未知模式 '%s'，可用模式：tokens / ast / run\n", mode);
    return 2;
}
```

### 6.4 插入位置说明

本步是起点，文件是**从上到下按下面这个顺序**写出来的（这个顺序就是 C 编译器的依赖顺序）：

```text
① 文件头注释
② #include
③ #define
④ TokType 枚举
⑤ Token 结构体
⑥ 全局变量
⑦ 错误处理函数（error_at / fatal_at）  ← 后面到处都要用，先定义
⑧ is_id_start / is_id_char
⑨ add_token
⑩ lex()
⑪ type_name()
⑫ read_file()
⑬ print_escaped()
⑭ 三个模式函数
⑮ main()
```

**为什么错误处理函数要放在 `lex()` 前面？**
C 语言要求"先声明后使用"。如果 `lex()` 里调用了 `fatal_at()`，
而 `fatal_at` 定义在文件更后面，编译会报
`implicit declaration of function 'fatal_at'`（C11 下这是错误，不是警告）。
当然你也可以在文件顶部加一堆前向声明（prototype），但**直接把定义放前面更简单**。

同理，`type_name()` 在 `mode_tokens()` 之前、`read_file()` 在 `mode_tokens()` 之前，
都是同一个道理。

## 7. 测试

`./run.sh` 的真实输出：

```text
▶ 用例（模式：tokens）
  ✅ case_01_numbers.ml           [tokens]
  ✅ case_02_keywords.ml          [tokens]
  ✅ case_03_strings.ml           [tokens]
  ✅ case_04_comments.ml          [tokens]
  ✅ case_05_operators.ml         [tokens]
  ✅ case_06_bad_char.ml          [tokens]
  ✅ case_07_unterminated_string.ml [tokens]
▶ demo.ml
  | [2:1] LET      let
  | [2:5] ID       x
  | [2:7] ASSIGN   = 
  | [2:9] NUM      10  (10)
  | [2:11] SEMI     ;
  | 
  | [3:1] LET      let
  | [3:5] ID       y
  | [3:7] ASSIGN   = 
  | [3:9] NUM      3.14  (3.14)
  | [3:13] SEMI     ;
  | 
  | [4:1] LET      let
  | [4:5] ID       name
  | [4:10] ASSIGN   = 
  | [4:12] STR      "MiniLang"
  | [4:22] SEMI     ;
  | 
  | [6:1] FN       fn
  | [6:4] ID       add
  | [6:7] LP       (a
  | [6:8] ID       a
  | [6:9] COMMA    , 
  | [6:11] ID       b
  | [6:12] RP       ) 
  | [6:14] LB       { 
  | [6:16] RETURN   return
  | [6:23] ID       a
  | [6:25] PLUS     + 
  | [6:27] ID       b
  | [6:28] SEMI     ; 
  | [6:30] RB       }
  | 
  | [7:1] IF       if
  | [7:4] LP       (x
  | [7:5] ID       x
  | [7:7] GT       > 
  | [7:9] NUM      1  (1)
  | [7:10] RP       ) 
  | [7:12] LB       { 
  | [7:14] ID       print
  | [7:19] LP       ("
  | [7:20] STR      "big\n"
  | [7:27] RP       );
  | [7:28] SEMI     ; 
  | [7:30] RB       } 
  | [7:32] ELSE     else
  | [7:37] LB       { 
  | [7:39] ID       print
  | [7:44] LP       ("
  | [7:45] STR      "small\n"
  | [7:54] RP       );
  | [7:55] SEMI     ; 
  | [7:57] RB       }
  | 
  | [8:1] FOR      for
  | [8:5] LP       (l
  | [8:6] LET      let
  | [8:10] ID       i
  | [8:12] ASSIGN   = 
  | [8:14] NUM      0  (0)
  | [8:15] SEMI     ; 
  | [8:17] ID       i
  | [8:19] LT       < 
  | [8:21] NUM      3  (3)
  | [8:22] SEMI     ; 
  | [8:24] ID       i
  | [8:26] ASSIGN   = 
  | [8:28] ID       i
  | [8:30] PLUS     + 
  | [8:32] NUM      1  (1)
  | [8:33] RP       ) 
  | [8:35] LB       { 
  | [8:37] ID       print
  | [8:42] LP       ("
  | [8:43] STR      "{i}\n"
  | [8:50] RP       );
  | [8:51] SEMI     ; 
  | [8:53] RB       }
  | 
  | [9:1] EOF      <文件结束>

✅ 全部通过（8 项）
```

### 逐个用例说明

| 用例 | 输入 | 期望 | 为什么是这个结果 |
|---|---|---|---|
| `case_01_numbers.ml` | `let x = 10;` / `let y = 3.14;` | stdout 见下 | `let` 是关键字 → `T_LET`；`x` 是标识符 → `T_ID`；`10` 是数字 → `T_NUM` 且额外打印 `(10)`（`%g` 格式） |
| `case_02_keywords.ml` | `let fn return if else for` | 6 个关键字 token | 证明"先当标识符再查表"这条路走通了：每个词都被查表改成了对应的类型 |
| `case_03_strings.ml` | `print("hello\n");` | `STR` token 内容是 `"hello\n"` | 字符串里的 `\n` 被解转义成真的换行，打印时 `print_escaped()` 又显示回 `\n`。注意 `LP` 那一行末尾是 `("`——`(` 是括号 token，字符串是另一个 token |
| `case_04_comments.ml` | 整行注释 + 行尾注释 | 只剩 4 个 token | 注释被完全跳过；但行号**没有**被压缩——第 2 行的 `let` 依然报告 `[2:1]`，说明注释"占位置" |
| `case_05_operators.ml` | `a == b != c <= d >= e && f || g ! h` | 12 个 token | 双字符运算符全部被正确识别。`!` 单独出现是 `T_NOT`，不是 `T_NE` |
| `case_06_bad_char.ml` | `let x = 1 @ 2;` | stderr `错误 [1:11]: 无法识别字符 '@'`，退出码 1 | `@` 不在任何一个分支里，走到 `switch` 的 `default`，`fatal_at` 报错并 `exit(1)`。列号 11 是 `@` 的位置 |
| `case_07_unterminated_string.ml` | `let s = "abc` | stderr `错误 [1:9]: 字符串不能直接跨行，请使用 \n`，退出码 1 | 字符串扫到 `\n`（文件结尾前的换行）时触发"跨行"检查。注意列号 9 是**字符串开引号**的位置，不是出错那一行的位置 |

`case_01_numbers.stdout` 的完整内容：

```text
[1:1] LET      let
[1:5] ID       x
[1:7] ASSIGN   = 
[1:9] NUM      10  (10)
[1:11] SEMI     ;

[2:1] LET      let
[2:5] ID       y
[2:7] ASSIGN   = 
[2:9] NUM      3.14  (3.14)
[2:13] SEMI     ;

[3:1] EOF      <文件结束>
```

> 末尾那个 `EOF` token 的行号是 `[3:1]`：因为 `case_01_numbers.ml` 的最后一行是
> `let y = 3.14;` 后面还有一个换行符，扫描器消费掉换行后 `src_line` 变成 3、`src_col` 归 1，
> 然后循环遇到 `'\0'` 退出，添加 EOF token。
> **EOF token 的行列号 = 源码结束处的位置**，这在第 12 步做错误定位时很有用。

### 其他真实运行

**未知模式**（退出码 2）：

```console
$ ./minilang demo.ml foo
未知模式 'foo'，可用模式：tokens / ast / run
```

**参数个数不对**（退出码 2）：

```console
$ ./minilang
用法: ./minilang 脚本.ml [tokens|ast|run]
示例: fn square(x) { return x * x; } print("{square(5)}\n");
```

**还没实现的模式**（退出码 1）：

```console
$ ./minilang demo.ml ast
语法分析尚未实现，请先用 tokens 模式查看词法结果

$ ./minilang demo.ml
解释执行尚未实现，请先用 tokens 模式查看词法结果
```

注意 `ast` / `run` 这两条提示是打到 **stderr** 的，所以即使你把 stdout 重定向到文件，
提示依然会出现在终端上——这是故意的：它们是**错误**，不是正常输出。

## 8. 易错点与坑

### 坑 1：双字符运算符写在了单字符后面

**现象**：`a == b` 被切成 `a` `=` `=` `b` 四个 token。

**原因**：违反了"最长匹配"原则。`=` 的分支先命中了。

**正确写法**：双字符的 `if` 链必须整体放在 `switch (c)` **之前**，
并且双字符分支命中后要 `i += 2; src_col += 2;` 然后 `continue`。

**怎么验证**：`./minilang tests/case_05_operators.ml tokens`，应该看到 `EQ` 而不是两个 `ASSIGN`。

### 坑 2：`isdigit` 传了负数

**现象**：源码里有中文时，程序可能行为异常，或者用 `-fsanitize=undefined` 编译时报警：

```text
runtime error: argument to 'isdigit' is not in the range of representable values of 'unsigned char'
```

**原因**：`char` 在 x86 上默认有符号，UTF-8 中文的第一个字节是 `0xE4`（= 228），
赋给 `char` 就变成 `-28`。把 `-28` 传给 `isdigit` 是未定义行为。

**正确写法**：所有 `<ctype.h>` 函数的参数都要先 `(unsigned char)` 转换。
本步的正确姿势是：

```c
int c = (unsigned char)src[i];        /* 只转一次，后面都用 c */
if (isdigit(c) || (c == '.' && isdigit((unsigned char)src[i+1]))) {
```

### 坑 3：`strtod` 不检查 `end` 指针

**现象**：`strtod("12abc", &end)` 返回 `12.0`，`"abc"` 被静默丢掉，没有任何报错。

**原因**：`strtod` 遇到无法解析的字符就停下，这是它"尽力而为"的语义。

**正确写法**：`if (!end || *end) fatal_at(...)`。

**怎么验证**：临时把 `lex()` 里的数字扫描改成只吃数字（不吃小数点），
然后输入 `1.5`，`end` 就会指向 `'.'`，报错"非法数字 '1'"。

### 坑 4：`snprintf` 的第二个参数写成字面量

**现象**：编译通过，但以后改了 `MAX_TEXT` 就会越界。

**原因**：`snprintf(dst, size, ...)` 的第二个参数是**目标缓冲区大小**，
不是"格式串里 `%s` 的宽度"。

**正确写法**：`snprintf(t->text, sizeof(t->text), "%s", text ? text : "")`。
永远用 `sizeof(目标数组)`，不要写魔法数字。

### 坑 5：`add_token` 的上限检查写成 `>`

**现象**：token 数正好等于 `MAX_TOKENS` 时越界写入。

**原因**：`ntok` 是"下一个空位的下标"。当 `ntok == MAX_TOKENS` 时数组已经满了。

**正确写法**：`if (ntok >= MAX_TOKENS)`。

### 坑 6：`src[i+1]` 在字符串末尾越界

**现象**：源码最后一个字符是 `+` 时，`src[i+1]` 读到的是 `'\0'`（合法，因为 `read_file`
多分配了一个字节并写了 `b[got] = 0`），但如果你自己写 `fgets` 就不一定有这层保护。

**原因**：`src` 是以 `\0` 结尾的，`src[len]` 合法，`src[len+1]` 越界。

**正确写法**：本步的 `read_file` 显式 `malloc(len + 1)` 并置 `b[got] = 0`，
所以 `src[i+1]` 最多读到那个 `'\0'`，安全。
**但** `two[1] = src[i+1]` 读到的 `'\0'` 参与 `strncmp` 比较不会匹配任何运算符，
行为正确。如果你改用 `strncmp(two, "==", 2)` 但 `two` 没初始化，
就会拿栈上的垃圾字节去比——这是 001 这类缓冲区 bug 的经典来源。

### 坑 7：`fatal_at` 里忘了 `exit(1)`

**现象**：程序打完错误信息后继续往下跑，可能越界或者输出乱七八糟的东西。

**原因**：`fatal_at` 名字里有 "fatal"（致命），语义上就该终止。

**正确写法**：`exit(1)`。同时用 `_Noreturn` 标注，让编译器帮你检查。

### 坑 8：`mode_ast` / `mode_run` 里的 `(void)path`

**现象**：编译器报 `unused parameter 'path'` 警告。

**原因**：`-Wall -Wextra` 会对未使用的参数报警。

**正确写法**：`(void)path;` 显式表示"我知道它没用，我故意的"。
第 2、3 步真正用上 `path` 时，把 `(void)path;` 删掉就行。

## 9. 自测题

### 9.1 概念题

**Q1.** 为什么 `Token` 结构体里要存 `line` 和 `col`，尽管第 1 步一个错误都还没报？

<details>
<summary>答案</summary>

因为**错误定位**需要它们。第 2 步的语法错误、第 5 步的"未定义变量"、
第 12 步的"出错源码行 + `^` 指示符"，全都依赖行列号。

第 1 步省掉它们，第 12 步想加定位就得重写词法分析器（几十处改动）；
第 1 步加它们，第 12 步只是多存一个 `src_line` 数组而已。

这是本教程的第一条工程经验：**接口定型要一次到位，后面只增不改。**
</details>

**Q2.** 为什么 `==` 的匹配必须放在 `=` 之前？

<details>
<summary>答案</summary>

最长匹配原则（maximal munch）：token 匹配总是取能匹配的最长的那一个。

如果 `=` 先命中，`a == b` 会变成 `a` `=` `=` `b` 四个 token，
语法上完全说不通，而且这个 bug **不会导致编译错误**，只会在运行时表现得很怪。
</details>

**Q3.** 关键字为什么"先当标识符再查表"，而不是写一个 `is_keyword()` 先判断？

<details>
<summary>答案</summary>

因为关键字的**词法形态和标识符完全一样**（字母开头、字母数字下划线组成）。
判断"这是不是关键字"必须先把整个词读出来。所以正确顺序是：
**先按标识符规则扫完 → 拿到字符串 → 查表 → 是关键字就改 token 类型。**

先写 `is_keyword()` 的话，你还是得先扫完整个词才能判断，那一遍扫描就白做了。
</details>

**Q4.** `error_at` 和 `fatal_at` 有什么区别？为什么第 1 步就要定好它们？

<details>
<summary>答案</summary>

- `error_at(line, col, fmt, ...)`：打印错误、置 `had_error = 1`，**继续执行**。
- `fatal_at(Token t, fmt, ...)`：打印错误、`exit(1)`，**立刻终止**。

签名（第 1 步定，之后 12 步不改）是 INV-7。
本步 `fatal_at` 收 `Token` 而不是 `(line, col)`，是因为语法分析器手里拿到的就是 token，
整个传进去省得拆。
</details>

### 9.2 预测题（改一行会发生什么）

**Q1.** 把 `if (isdigit(c) || (c == '.' && isdigit((unsigned char)src[i+1])))` 里的
`(c == '.' && ...)` 删掉，会发生什么？

<details>
<summary>答案</summary>

`.5` 不再被识别成数字。`.` 会掉到 `switch` 的 `default` 分支，报
`错误 [1:9]: 无法识别字符 '.'` 并退出。

（验证：`printf 'let y = .5;\n' > /tmp/t.ml && ./minilang /tmp/t.ml tokens`）
</details>

**Q2.** 把 `if (ty != T_EOF) { i += 2; ... }` 里的 `i += 2` 改成 `i += 1`，会发生什么？

<details>
<summary>答案</summary>

双字符运算符只吃掉第一个字符。`a == b` 会变成 `a` `==` `=` `b`——
`==` 被识别了，但第二个 `=` 又被单独识别成 `T_ASSIGN`。

列号也会错位：双字符分支只 `src_col += 1`，后面所有 token 的列号都会偏小 1。
</details>

**Q3.** 把 `Token` 结构体里的 `int line, col;` 删掉，程序还能编译吗？

<details>
<summary>答案</summary>

不能。`add_token` 里有 `t->line = line; t->col = col;`，
`fatal_at` 里也有 `t.line` / `t.col`，全是编译错误。

这说明行列号在第 1 步**已经被用到了**（不只是"预留"）——
词法分析器自己的错误（无法识别字符、字符串跨行）就靠它们定位。
</details>

**Q4.** 把 `while (src[i] && src[i] != '\n')` 里的 `src[i] &&` 删掉，会发生什么？

<details>
<summary>答案</summary>

如果文件最后一行是注释且**没有结尾换行符**，`src[i]` 会越过 `'\0'` 继续读，
直到在内存里碰巧遇到一个 `'\0'`。这是**缓冲区越界读**，可能读到无关内存，
也可能崩溃。

`src[i] &&` 这个检查保证循环在 `'\0'` 处停下。
</details>

### 9.3 动手题

**Q1.** 新增一个关键字 `while`，让它能被 `tokens` 模式识别为 `T_WHILE`。
最少需要改几处？

<details>
<summary>答案</summary>

三处：
1. `TokType` 枚举里加 `T_WHILE`；
2. `lex()` 的查表链里加 `else if (!strcmp(b, "while")) ty = T_WHILE;`；
3. `type_name()` 里加 `case T_WHILE: return "WHILE";`。

验证：

```sh
printf 'while\n' > /tmp/w.ml
./minilang /tmp/w.ml tokens
```

应该输出 `[1:1] WHILE   while`。

（注意：加完之后 `while` 就成了保留字，不能再当变量名用了。）
</details>

**Q2.** 让词法分析器支持 `/* ... */` 块注释（可以跨行）。

<details>
<summary>答案</summary>

思路：在 `//` 注释那个分支**之后**加一个新分支：

```c
/* 块注释：/* ... */ */
if (c == '/' && src[i+1] == '*') {
    i += 2; src_col += 2;
    while (!(src[i] == '*' && src[i+1] == '/')) {
        if (!src[i]) fatal_at((Token){.line=line, .col=col}, "块注释没有结束");
        if (src[i] == '\n') { src_line++; src_col = 1; }
        else src_col++;
        i++;
    }
    i += 2; src_col += 2;
    continue;
}
```

**位置很关键**：必须在 `//` 那个分支**之后**、数字分支**之前**——
因为两者都以 `/` 开头，必须先试长的。

**坑**：`/*` 必须放在 `//` 之后还是之前？两者首字符都是 `/`，
但 `//` 是 `/*` 的前缀冲突吗？不是——它们第二字符不同，
所以谁先谁后都行，**只要都在单字符 `/` 的 `switch` 之前**。
不过为了可读性，建议把块注释放在行注释后面。

验证：

```sh
printf 'let /* 中间注释 */ x = 1;\n' > /tmp/bc.ml
./minilang /tmp/bc.ml tokens
```

应该得到 `[1:1] LET let` / `[1:20] ID x` / ...
</details>

**Q3.** 让 `tokens` 模式在打印数字时**多打一位精度**，
比如 `3.14` 打成 `3.14000000000000`（即 `%g` 改成 `%.15f`）。

<details>
<summary>答案</summary>

把 `mode_tokens` 里的：

```c
if (t->type == T_NUM) printf("  (%g)", t->num);
```

改成：

```c
if (t->type == T_NUM) printf("  (%.15f)", t->num);
```

**注意**：这样会破坏 `tests/case_01_numbers.stdout` 和 `case_05` 之外的期望输出，
`./run.sh` 会失败——这是**预期内的**，因为你在改输出的格式。
改完记得把 `.stdout` 文件一起更新，或者改完就改回去。

（顺便体会一下：`%g` 会自动去掉多余的 0 和小数点，`%f` 不会。
本教程后面统一用 `%g`，因为 `15` 应该显示成 `15` 而不是 `15.000000`。）
</details>

**Q4.**（进阶）给 `tokens` 模式加一个统计功能：
在 token 列表**前面**先打一行 `共 N 个 token`。

<details>
<summary>答案</summary>

在 `mode_tokens` 的循环之前加一句：

```c
printf("共 %d 个 token\n", ntok);
```

注意 `ntok` **包括最后的 EOF token**，所以数字会比"真正的 token 数"多 1。
如果你觉得应该不算 EOF，可以写 `ntok - 1`。

**坑**：`printf` 输出的顺序和 `fprintf(stderr, ...)` 混在一起时，
因为 stdout 是**块缓冲**、stderr 是**无缓冲**，
终端上可能看到 stderr 的内容跑到前面去。
这是 C 的 I/O 缓冲策略导致的，不是 bug。
</details>

## 10. 本步的设计取舍

**为什么把词法分析器一次做全（INV-1）？**

因为词法分析器是整个解释器里**最稳定**的一层——语言的"词汇表"不会随功能增长而变。
反而是语法分析和求值层会不断长出东西。如果边做边加 token，
每加一个都要改 `TokType` 枚举、`type_name()`、`lex()` 三处，
diff 会变得又碎又吵，而且容易漏改一处导致 `type_name()` 返回 `"?"`。

一次性认全 31 个 token 的代价是：第 1 步的代码里有几个 token
（`T_AND`、`T_OR`、`T_NOT`、`T_FN`…）在接下来好几步里都用不上。
**这是"提前付出、后面免债"的典型取舍**：现在多写 10 行，
省掉的是后面 12 步里可能出现的几十次三处同改。

**为什么 `Token.text` 用 4096 字节的固定数组，而不是指针？**

指针方案（`const char *text` 指向源文件里的原始位置）更省内存，
但会带来三个麻烦：字符串 token 需要单独存一份（因为要解转义），
于是就变成"指针 + 长度 + 可选的独立副本"；
而且指针指向 `src`，`src` 被 `free` 之后 token 就成了悬垂指针。

固定数组的代价是 `20000 × 4096 ≈ 82 MB` 的静态内存。
在教学项目里这个代价可以接受（BSS 段不占磁盘，操作系统按需分页），
换来的是"token 自包含、不依赖源文件生命周期"这个巨大的简化。

**还没做什么？**

- 不支持 `/* */` 块注释。
- 不支持科学计数法（`1e5`）、十六进制（`0x1F`）、数字分隔符（`1_000`）。
- 不支持字符串内的花括号插值（那是第 9 步的事）。
- 不支持 `++` / `--`（第 7 步的 `for` 更新部分也只支持 `i = 表达式`）。
- 错误信息只有行列号，还不能显示源码行和 `^`（第 12 步才加）。
- 词法分析是"一次扫完整个文件"，不是流式的。对教学项目来说，
  一次性 `read_file` + 全量 token 数组是最简单的做法。

## 11. 下一步预告

现在你有了 token 流，但还不会"理解"它们。
下一步我们写**递归下降语法分析器**，把 token 流变成**抽象语法树（AST, Abstract Syntax Tree）**——
一棵用 `Node` 结构体搭出来的树，它会明确告诉你 `1 + 2 * 3` 到底该先算哪一步。

你会学到：
- 为什么需要 AST，而不是"边解析边执行"；
- `Node` 结构体每个字段的用途；
- 递归下降怎么用"一层一个函数"表达运算符优先级；
- 一个通用的 `bin_level` 循环怎么把整张优先级表变成代码。

第 2 步的 `ast` 模式会真正跑起来，输出带 `├──` `└──` 连接线的语法树。

→ [第 2 步 · 表达式语法分析 → AST](../002/README.md)
