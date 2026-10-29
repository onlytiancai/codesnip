# 第 9 步 · print 与字符串模板

> 进度 █████████░ 9/13 ｜ 上一节：[第 8 步 · 内置函数](../008/README.md) ｜ 下一节：[第 10 步 · 用户函数：定义与调用](../010/README.md)

## 1. 本步目标

读完你将能够：

- 让 `print("x = {x}\n")` 把变量 `x` 的值嵌进字符串里，输出 `x = 10`。
- 说清楚**为什么 `let s = "hi";` 会报错**（"变量目前只支持数字类型"），并解释 `Value` 里那个 `is_string` 标记位存在的意义。
- 解释 `\n` `\t` `\r` `\"` `\\` 这些转义是在**词法分析阶段**解好的——第 1 步 `tokens` 模式打印出来的字符串就已经是解码后的内容。
- 解释模板**只支持 `{标识符}`**：`"{sqrt(81)}"` 会被当成"名为 `sqrt(81)` 的变量"而报"未定义变量"，必须先赋值给变量再打印。
- 指出本步是整套教程里**少见的、需要修改既有行**的地方（白名单 W-1 / W-2），并说明为什么改在这里是合理的。

## 2. 运行方式

```sh
./run.sh                 # 编译 + 跑全部用例 + 跑 demo
./minilang demo.ml       # 默认 run 模式
./minilang demo.ml tokens
./minilang demo.ml ast
```

## 3. 现在的程序能做什么

| 模式 | 输入 | 输出 | 退出码 |
|---|---|---|---|
| `tokens` | 任意 `.ml` | token 列表（`STR` 里的转义已经解码） | 0（有错为 1） |
| `ast` | 任意 `.ml` | 语法树，字符串字面量显示为 `字符串 "..."` | 0（有错为 1） |
| `run`（默认） | 任意 `.ml` | 真正执行程序；`print` 能渲染 `{变量}` 模板 | 0；语法/运行错误为 1 |

第 8 步已经支持 `abs`/`round`/`sqrt`/`floor`/`ceil`/`pow`/`min`/`max`；本步**只增加了字符串模板**这一件事，别的语法一行没动。

`009/demo.ml` 的内容：

```
// 第 9 步：字符串模板
let x = 10;
let y = 32;
print("x = {x}, y = {y}\n");
print("x + y = {x}\n");
```

真实运行结果：

```sh
$ ./minilang demo.ml
x = 10, y = 32
x + y = 10
```

注意最后一行输出的是 **10** 而不是 42——模板里只做**变量替换**，不做算术。`{x + y}` 会报"未定义变量 'x + y'"。

用 `tokens` 模式看字符串在词法分析之后长什么样：

```sh
$ cat /tmp/tok.ml
print("a\tb\n");
$ ./minilang /tmp/tok.ml tokens
[1:1] ID       print
[1:6] LP       ("
[1:7] STR      "a\tb\n"
[1:15] RP      );
[1:16] SEMI    ;

[2:1] EOF      <文件结束>
```

`STR` 里显示的 `"a\tb\n"` 是 `print_escaped()` 重新转义之后的**显示形式**；token 内部存的已经是**真的制表符和真的换行符**了。详见第 5.5 节。

## 4. 前置知识回顾

**（1）`Value` 里早就有字符串的位置。** 第 3 步定型了这个结构体：

```c
typedef struct {
    double value;             /* 数值            */
    int    is_string;         /* 1 表示是字符串  */
    char   string[MAX_TEXT];  /* 字符串内容      */
} Value;
```

当时 `is_string` 和 `string` 两个字段**一个字节都没用过**，源码里那行注释写得很清楚："`is_string` 和 `string` 这两个字段现在还没用上，是提前留好的。" 本步就是它们上岗的日子。**这就是 INV-2（第 3 步把 `Value` 定型成"数字或字符串"）兑现价值的地方**——因为第 3 步就把 `string` 字段预留好了，本步加字符串时**一个函数签名都不用改**。

**（2）`print` 从一开始就是个"特例"，不是普通函数。** 第 8 步引入内置函数表（`Builtin`）的时候，源码注释里专门写了一句：

> 表里放的是"数字进、数字出"的纯函数。`print` 不在表里——它要输出，而且下一步还要访问作用域去查模板里的变量，需求不一样。

本步就是把那句"下一步"变成现实。

**（3）`eval()` 里 `case N_STR:` 一直是个"占位符"。** 第 3 步写的是：

```c
case N_STR:
    fatal_at(n->tok, "求值器还不支持字符串，字符串会在讲 print 模板时支持");
    break;
```

这句话里的"会在讲 print 模板时支持"，指的就是现在。

## 5. 本步引入的核心概念

### 5.1 概念一：字符串是**字面量**，不能存进变量

**问题**：为什么 `let s = "hi";` 会失败？

**思路**：MiniLang 的变量系统从第 5 步开始就只支持数字。这不是偷懒，而是**刻意的简化**：`Env` 的变量表存的是 `double value`：

```c
typedef struct {
    char   name[MAX_NAME];
    double value;
} Var;
```

变量表里根本没有"字符串"这一列。要支持 `let s = "hi";`，就得把 `Var.value` 改成 `Value`，整条赋值/查找链路全部要改一遍，还要处理"字符串变量能不能相加"这种问题。**教学版不做这件事。**

于是 `let s = "hi";` 会走到 `exec()` 的 `case N_LET:`：

```c
case N_LET: {
    Value v = eval(n->a, e);
    if (v.is_string) fatal_at(n->tok, "变量目前只支持数字类型");
    define_var(e, n->tok, n->text, v.value);
    return f;
}
```

**这行 `is_string` 检查在第 8 步就已经存在了**（当时是死代码，因为 `eval` 遇到字符串会先 `fatal_at` 掉）。本步把 `eval` 改成真的会返回字符串值之后，这行检查**自动就生效了**——这就是 INV-8（只加不改）的红利：一行都没改，行为却变了。

真实运行：

```sh
$ cat /tmp/letstr.ml
let s = "hi";
$ ./minilang /tmp/letstr.ml
错误 [1:1]: 变量目前只支持数字类型
```

**代码要点**：`fatal_at(n->tok, ...)` 用的是 `N_LET` 节点的 token，也就是 `let` 关键字的位置（`[1:1]`），而不是字符串字面量的位置（`[1:9]`）。语义上"变量不能是字符串"是 `let` 语句本身的毛病，报在 `let` 上更准确。

**边界情况**：那 `print` 的参数为什么可以是字符串？因为 `print` 走的是 `eval` 里**特殊分支**，它拿到的 `Value` 直接拿去渲染模板，**从不进变量表**。所以"字符串是字面量"这个限制，对 `print` 是不生效的。

```
字符串的生命周期：

  "hi"  ──►  eval(N_STR)  ──►  Value{is_string=1, string="hi"}
                                             │
                        ┌────────────────────┴────────────────────┐
                        │                                         │
                  print("hi")                              let s = "hi"
                        │                                         │
                        ▼                                         ▼
              print_template() 渲染输出              fatal_at("变量目前只支持数字类型")
              （不落地，栈上临时值）                    （想落地 → 变量表放不下）
```

### 5.2 概念二：`is_string` 这个标记位为什么必须存在

**问题**：`Value` 里已经有 `double value` 了，为什么还要额外加一个 `int is_string`？直接看 `value` 是不是 0 来判断不就行了吗？

**思路**：不行，因为 **`0` 是一个完全合法的字符串**：

```
print("");        ← 打印一个空字符串
```

如果用 `value == 0` 当"这是字符串"的判据，那空字符串就会被当成数字 0 处理。更根本的问题是：一个 `Value` 里 `value` 和 `string` **两个字段同时有效**时，无法判断该读哪个。

所以 `Value` 用了**"标签 + 联合体"**的思路（没用真正的 `union`，是为了教学上简单、也为了避免类型双关的坑）：

```
Value 的"类型系统"：

   is_string = 0 （数字）              is_string = 1 （字符串）
   ┌───────────────────────┐          ┌───────────────────────┐
   │ value  = 42.0        │  ← 读这个│ value  = 0            │  ← 忽略
   │ string = ""（不用管） │          │ string = "hello"      │  ← 读这个
   └───────────────────────┘          └───────────────────────┘
```

**代码要点**：`is_string` 是 `int` 而不是 `_Bool`，这是 C 里常见的教学选择（`_Bool` 更精确，但 `int` 在 `printf("%d")` 和比较时更省事）。构造一个数字值必须用工厂函数 `number()`：

```c
static Value number(double x) {
    Value v;
    v.value = x;
    v.is_string = 0;
    v.string[0] = '\0';
    return v;
}
```

**三个字段一个都不能漏初始化**——`number()` 里 `v.string[0] = '\0';` 那行看着多余，其实是防御：万一有人把 `Value` 直接 `printf("%s", v.string)` 打印，未初始化的 `string` 数组就是一段垃圾字节。

**边界情况**：本步的 `N_STR` 分支是**手写**的，没有调用工厂函数：

```c
case N_STR: {
    Value v;
    v.value = 0;
    v.is_string = 1;
    snprintf(v.string, sizeof(v.string), "%s", n->text);
    return v;
}
```

这里 `v.value = 0` 是必要的（不能留垃圾值），但 `v.string` 是被 `snprintf` **完整覆盖**的，所以不用逐字节清零。这是 C 里"要么全赋值、要么 memset"的常见取舍。

### 5.3 概念三：白名单 W-1 / W-2——本步是"改既有行"的地方

整套教程有一条铁律（INV-8）：**除白名单外，新代码只插入、不修改既有行**。所以第 1 步到第 8 步的 diff 全是"新增"，第 12 步的 diff 也全是"新增"。

**本步是白名单里明确列出的两处**：

| 编号 | 改了什么 | 为什么必须改 |
|---|---|---|
| **W-1** | `eval` 里 `print` 的处理：加一个 `if (v.is_string)` 分支 | 第 8 步 `print` 只有一条"打印数字"的路径。不加分支，字符串永远走不到模板 |
| **W-2** | `N_UNARY` 加一行 `is_string` 检查；`N_BINARY` 加两行 `is_string` 检查 | 第 8 步 `eval` 里 `a.value` / `x.value` / `y.value` 是直接当数字用的。字符串进来会静默变成 0，静默出错比报错更可怕 |

**为什么"改在这里"是合理的？**

先说 W-2 为什么**必须**改。第 8 步的 `N_BINARY` 长这样：

```c
case N_BINARY: {
    Value x = eval(n->a, e);
    ...
    Value y = eval(n->b, e);
    double a = x.value, b = y.value;      /* ← 字符串的 value 是 0，直接被当成数字 0 */
    switch (n->op) {
        case T_PLUS: return number(a + b);
```

本步 `eval` 真的会返回字符串值了。如果不加检查：

```
let s = "hi";
let n = s + 1;      →  0 + 1 = 1     （不报错，静悄悄算错）
```

**静默的错误结果比响亮的报错糟糕得多。** 一个写错了程序的作者，看到 `n == 1` 会以为自己的逻辑是对的，然后在别处找半天 bug。所以类型检查必须加。

再说 W-1 为什么**必须**改。第 8 步的 `print`：

```c
Value v = eval(n->items[0], e);
if (v.is_string)
    fatal_at(n->tok, "print 的参数必须是数字");
printf("%.15g", v.value);
```

本步 `eval` 会返回字符串了，这段代码就会对每个 `print("...")` 报错。**报错信息本身是对的**（那时候确实不支持），但现在要支持了，就得把"数字路径"和"字符串路径"分开。

**为什么不能"另起炉灶"？** 比如把 `print` 从 `eval` 的 `N_CALL` 里挪走、单独在 `exec` 里处理——那样会破坏 INV-4（`eval` 管表达式、`exec` 管语句的物理分离），也会让 `print` 不能再出现在表达式中间（比如 `let x = print(1);`）。**在既有分支里加一个 `if`，是改动最小的做法。**

**顺便说一句**：本步的 diff 里还改了一句**注释**：

```diff
-            /* print 特殊处理：它要输出，而且下一步还要访问作用域查模板变量 */
+            /* print 特殊处理：它要输出，而且还要访问作用域查模板里的变量 */
```

第 8 步那句注释里的"下一步"现在变成"现在"了。改注释不算 INV-8 意义上的"修改既有逻辑"，但它确实让 diff 里多了一行噪音——**这是本步唯一一处可以避免的 diff 噪音**，如果你想让 diff 更干净，保持原注释不动也完全没问题。

### 5.4 概念四：模板的**逐字符扫描**

**问题**：`print("x = {x}, y = {y}\n")` 怎么把 `{x}` 换成 `10`？

**思路**：模板的语法极简——**只有一种占位符**：`{标识符}`。所以不需要真正的解析器，一个 `while` 循环 + 一个 `strchr` 就够了。

`print_template()` 的完整状态机：

```
                    ┌─────────────────────────┐
        ┌──────────►│  普通字符               │
        │           │  putchar(*s); s++;      │
        │           └───────────┬─────────────┘
        │                       │ 遇到 '{'
        │                       ▼
        │           ┌─────────────────────────┐
        │           │  strchr(s+1, '}')       │
        │           │  找配对的 '}'           │
        │           └───────────┬─────────────┘
        │                       │
        │          ┌────────────┴────────────┐
        │          │ 找到了                   │ 没找到
        │          ▼                         ▼
        │   ┌──────────────────────┐  ┌──────────────────────┐
        │   │ 取出变量名 name      │  │ 当成普通字符输出     │
        │   │ get_var(e,tok,name)  │  │ putchar('{'); s++;  │
        │   │ printf("%.15g", v)   │  └──────────────────────┘
        │   │ s = end + 1          │
        │   └──────────┬───────────┘
        │              │
        └──────────────┘
```

拿真实字符串走一遍（`s` 是扫描指针）：

```
源码（内存里的字符串）:  x = {x}, y = {y}\n
                          ↑         ↑
                     普通字符     普通字符

第 1 轮：s 指向 'x'      → 不是 '{' → 输出 'x'，s++
第 2 轮：s 指向 ' '      → 输出 ' '
第 3 轮：s 指向 '='      → 输出 '='
第 4 轮：s 指向 ' '      → 输出 ' '
第 5 轮：s 指向 '{'      → strchr(s+1,'}') 找到第 7 个字符
                        → name = "x"（长度 1）
                        → get_var(e, tok, "x") → 10
                        → 输出 "10"
                        → s = end + 1，跳到 ',' 之后
第 6 轮：s 指向 ','      → 输出 ','
...
```

**代码要点**：

- `const char *end = strchr(s + 1, '}');` —— 从 **`s + 1`** 开始找（跳过 `{` 自己），找到的第一个 `}` 就是配对的那个。
- `size_t len = (size_t)(end - (s + 1));` —— 指针相减得到的是**两个指针之间的元素个数**，正好是变量名的长度。
- `memcpy(name, s + 1, len); name[len] = '\0';` —— 复制到 `char name[MAX_NAME]` 里并**补上结尾的 `\0'`**。少了 `\0`，`get_var()` 里的 `strcmp` 会一路读越界。
- `s = end + 1; continue;` —— `continue` 让 `while` 直接进入下一轮，**跳过** `putchar(*s++)` 那一行。这一行如果忘了写，`'}'` 会被原样打出来。

**边界情况**：

- **没有配对的 `}`**：`strchr` 返回 `NULL`，代码**不报错**，而是把 `{` 当普通字符输出。`print("a{b");` 的输出是 `a{b`。
- **空模板 `{}`**：`len == 0` → 报"模板里的变量名无效"。
- **变量名太长**（≥ `MAX_NAME` = 128）：`len >= sizeof(name)` → 同样报"模板里的变量名无效"。注意判断是 `>=` 不是 `>`——`name` 只有 128 字节，能装的最长名字是 127 个字符加 `\0`。

**为什么用 `strchr` 而不是手写循环找 `}`？** 因为 `strchr` 在 C 标准库里、经过高度优化，而且写法短。代价是它**只能找第一个** `}`，所以 `{a}b}` 会被解析成"变量 a" + 剩余的 `b}`。这是"模板只支持标识符"的必然结果。

### 5.5 概念五：转义在**词法分析阶段**就解码了

**问题**：`print("a\n")` 里的 `\n` 到底是什么？是两个字符 `\` 和 `n`，还是一个换行符？

**思路**：看第 1 步的词法分析器怎么处理字符串字面量：

```c
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
    ...
    add_token(T_STR, b, 0, line, col);
}
```

**解码发生在 `add_token` 之前**。也就是说：

```
源码          "a\tb\n"
                ↓  词法分析（lex）
token.text    "a" + 真的 TAB + "b" + 真的 LF
                ↓  语法分析
N_STR.text    同上（snprintf 原样复制）
                ↓  求值
Value.string  同上
                ↓  print_template
putchar 逐字节输出
```

**这解释了一个第 1 步就能观察到的现象**：在 `tokens` 模式里，字符串 token 打印出来是 `"a\tb\n"`——因为 `mode_tokens()` 用 `print_escaped()` 把真的 TAB/LF **重新转义**成可读形式给你看。**显示时转义，存储时解码**，两件事分开做。

支持哪些转义？只有 5 个：

| 写法 | 解码成 |
|---|---|
| `\n` | 换行（LF，0x0A） |
| `\t` | 制表符（TAB，0x09） |
| `\r` | 回车（CR，0x0D） |
| `\"` | 双引号 |
| `\\` | 反斜杠 |

其它任何 `\x` 都会报错："不支持的字符串转义 \x"。

**顺便解决了一个 C 语言的经典坑**：`printf` 里 `%` 不能直接用于 `double`（要用 `fmod`），但字符串里的 `%` **完全没问题**——因为 `print_template` 是用 `putchar` 逐字节输出的，根本没走 `printf` 的格式解析。`print("100%%\n")` 会原样输出 `100%%`（MiniLang 没有字符串的 `%` 转义）。

### 5.6 概念六：`print` 不自动换行

**问题**：`print(42)` 输出什么？带换行吗？

**思路**：**不带**。这是从第 3 步就定下的行为，源码里那行注释说得很直白：

```c
printf("%.15g", v.value);   /* 数字之间不会自动换行 */
```

所以：

```
print(42); print(43);
```

输出是 `4243`（中间没有空格、没有换行）。

**这在本步正好被模板里的 `\n` 解决了**：

```
print("x = {x}\n");     ← 结尾的 \n 是字符串自带的换行
print("x = {x}");       ← 不写 \n 就没有换行
```

真实运行：

```sh
$ cat /tmp/nl.ml
print(42);
print(43);
$ ./minilang /tmp/nl.ml
4243
```

**代码要点**：数字用 `printf("%.15g", ...)`——`%.15g` 表示"用 `%g` 的格式，但保留 15 位有效数字"。所以 `15` 打成 `15`（不是 `15.000000`），`3.5` 打成 `3.5`，`1/3` 打成 `0.333333333333333`。

**边界情况**：模板里的变量也是用 `%.15g` 打的，所以 `print("{x}\n")` 里 `x = 42` 输出 `42`，`x = 42.5` 输出 `42.5`。

## 6. 动手：把代码加进来

### 6.1 改动清单

| 位置锚点 | 改了什么 | 大概多少行 | 为什么 |
|---|---|---|---|
| `set_var()` 函数之后、`Value` 结构体之前 | 新增 `print_template()` 函数 | 24 行 | 渲染 `{变量名}` 模板 |
| `eval()` 的 `case N_STR:` | 把 `fatal_at` 占位改成"返回一个字符串值" | 8 行（改 3 行） | 字符串终于有了值 |
| `eval()` 的 `case N_CALL:` 里 `print` 分支 | 加 `if (v.is_string)` 分支 | 8 行（改 4 行） | **白名单 W-1** |
| `eval()` 的 `case N_UNARY:` | 加一行 `is_string` 检查 | 1 行 | **白名单 W-2** |
| `eval()` 的 `case N_BINARY:` | 加两行 `is_string` 检查 | 3 行 | **白名单 W-2** |

合计 **约 44 行**，其中**新增 32 行、修改 9 行、修改注释 1 行**。这是整套教程里 diff 最"脏"的一步——但每一处修改都在白名单里，且理由充分。

### 6.2 分段讲解

#### 第 1 段：新增 `print_template()` 函数

```diff
+/* 打印字符串模板：把 {变量名} 换成变量的值。
+   逐字符扫描，遇到 '{' 就找配对的 '}'。 */
+static void print_template(const char *s, Env *e, Token tok) {
+    while (*s) {
+        if (s[0] == '{') {
+            const char *end = strchr(s + 1, '}');
+            if (end) {
+                size_t len = (size_t)(end - (s + 1));
+                char name[MAX_NAME];
+                if (len == 0 || len >= sizeof(name))
+                    fatal_at(tok, "模板里的变量名无效");
+                memcpy(name, s + 1, len);
+                name[len] = '\0';
+                /* 模板里的变量必须已经定义过 */
+                double v = get_var(e, tok, name);
+                printf("%.15g", v);
+                s = end + 1;
+                continue;
+            }
+        }
+        putchar(*s++);
+    }
+}
```

**逐行解释**：

- `static void print_template(const char *s, Env *e, Token tok)` —— `s` 是要渲染的字符串，`e` 是当前作用域（用来查变量），`tok` 是出错时用的位置信息。
- `while (*s)` —— 逐字符扫描直到字符串结尾（`'\0'`）。
- `if (s[0] == '{')` —— 注意这里写的是 `s[0]` 而不是 `*s == '{'`，两者等价，但 `s[0]` 更直白。
- `const char *end = strchr(s + 1, '}');` —— 找配对的 `}`。**从 `s + 1` 开始**，跳过 `{` 自己。
- `if (end)` —— 找到了才处理；没找到就落到最后的 `putchar(*s++)`，把 `{` 原样输出。
- `size_t len = (size_t)(end - (s + 1));` —— 指针相减。两指针相减的结果是**元素个数**（`ptrdiff_t`），这里变量名长度肯定不是负数，但养成 `(size_t)` 转换的习惯能避免某些编译器的警告。
- `char name[MAX_NAME];` —— 栈上的临时数组，`MAX_NAME` = 128。
- `if (len == 0 || len >= sizeof(name))` —— 两种非法情况：空名字，或者名字太长装不下。**用 `>=`** 是因为要留一个字节给 `\0`。
- `memcpy(name, s + 1, len); name[len] = '\0';` —— 复制并补结尾符。**`name[len] = '\0'` 这一行不能少**。
- `double v = get_var(e, tok, name);` —— 复用第 5 步就写好的 `get_var()`！它会沿作用域链往外找，找不到就报"未定义变量"。**这就是"模板里的变量必须已经定义过"的实现**。
- `printf("%.15g", v);` —— 数字用 `%.15g` 打印，和 `print(数字)` 完全一致。
- `s = end + 1; continue;` —— 跳过整个 `{...}`，从 `}` 之后继续。`continue` 是必须的，否则会再执行一次 `putchar(*s++)` 把 `'}'` 打印出来。

**如果你写错了会怎样**：

- 忘了 `name[len] = '\0';` → `get_var` 里的 `strcmp` 会越界读，行为不可预测（可能"能跑"，可能崩，可能报一个莫名其妙的变量名）。
- 忘了 `continue;` → 输出会多出一个 `}`：`print("{x}")` 会输出 `10}`。
- `strchr(s, '}')`（漏了 `+ 1`）→ 会把 `{` 本身当成变量名的一部分，`{x}` 会被当成变量名 `{x`，报"未定义变量 '{x'"。
- `get_var(e, tok, name)` 写成 `get_var(e, tok, s + 1)` → 传进去的是**没有结尾符的指针**，`strcmp` 会一路读到字符串末尾。
- `printf("%s", name)` 之类 → 变量名是数字，要用 `%.15g` 打印**值**，不是名字。

#### 第 2 段：`case N_STR:` 从"报错"改成"返回值"

```diff
-        case N_STR:
-            fatal_at(n->tok, "求值器还不支持字符串，字符串会在讲 print 模板时支持");
-            break;
+        case N_STR: {
+            /* 字符串字面量：求值结果是一个字符串值 */
+            Value v;
+            v.value = 0;
+            v.is_string = 1;
+            snprintf(v.string, sizeof(v.string), "%s", n->text);
+            return v;
+        }
```

**逐行解释**：

- `case N_STR: {` 后面加了花括号——因为里面要声明局部变量 `Value v`。C 的 `switch` 里所有 `case` 共享同一个块作用域，不加花括号会和别的 case 里的变量冲突。
- `v.value = 0;` —— 字符串没有数值，给 0。但**不能省**：否则 `v.value` 是垃圾值，将来谁读了它就出错。
- `v.is_string = 1;` —— **这一行是整个字符串支持的核心**。第 5.2 节说过，`is_string` 就是 `Value` 的"类型标签"。
- `snprintf(v.string, sizeof(v.string), "%s", n->text);` —— 把 token 里的字符串内容复制到 `Value` 里。**用 `snprintf` 而不是 `strcpy`**：`n->text` 和 `v.string` 都是 `char[MAX_TEXT]`，理论上等长，但 `snprintf` 会保证**最多写 `sizeof(v.string) - 1` 个字节并自动补 `\0`**。`strcpy` 在长度超出时会静默越界。
- `return v;` —— 返回一个字符串值。

**如果你写错了会怎样**：

- 忘了 `v.is_string = 1;` → `print("hi")` 会把字符串当数字，`v.value` 是 0，输出 `0`；`let s = "hi";` 也不会报错（因为 `is_string` 是 0），而是定义出一个值为 0 的变量。**这是最隐蔽的一种错**。
- 用 `strcpy(v.string, n->text)` → 本步两数组等长，暂时不会出事；但如果哪天 `MAX_TEXT` 和 `Token.text` 的长度不一样了，就会越界。**`snprintf` 是默认的安全选择**。
- 写成 `v.string = n->text;` → 编译报错：`array type 'char[4096]' is not assignable`。

#### 第 3 段：`print` 分支加字符串模板分支（白名单 W-1）

```diff
         case N_CALL: {
-            /* print 特殊处理：它要输出，而且下一步还要访问作用域查模板变量 */
+            /* print 特殊处理：它要输出，而且还要访问作用域查模板里的变量 */
             if (!strcmp(n->text, "print")) {
                 if (n->count != 1)
                     fatal_at(n->tok, "函数 'print' 参数数量错误：需要 1 个，实际 %d 个", n->count);
                 Value v = eval(n->items[0], e);
-                if (v.is_string)
-                    fatal_at(n->tok, "print 的参数必须是数字");
-                printf("%.15g", v.value);   /* 数字之间不会自动换行 */
+                if (v.is_string) {
+                    /* 字符串参数：当成模板渲染 */
+                    print_template(v.string, e, n->tok);
+                } else {
+                    printf("%.15g", v.value);   /* 数字之间不会自动换行 */
+                }
                 return number(0);
             }
```

**逐行解释**：

- `if (v.is_string) { ... } else { ... }` —— **两条路**：字符串走模板渲染，数字走 `%.15g`。数字那条路的代码**一个字没动**，只是被挪进了 `else` 分支。
- `print_template(v.string, e, n->tok)` —— 传进去的是 `v.string`（`Value` 里的字符串缓冲），不是 `n->items[0]->text`。因为参数可能不是字面量（比如 `print(x)` 里 `x` 是……等等，`let x = "hi"` 不允许，所以实际上 `print` 的字符串参数**目前只能是字面量**。但接口设计成"传字符串内容"更通用）。
- `n->tok` —— 出错时用的位置是 `print` 这个调用的位置（`[1:1]`），因为"模板里的变量没定义"是这次 `print` 调用的毛病。

**如果你写错了会怎样**：

- 忘了 `else` 分支（直接删掉原来的 `printf`）→ `print(42)` 会**什么都不输出**。
- 把 `print_template(v.string, ...)` 写成 `print_template(v.value, ...)` → 编译报错（`double` 不能传给 `const char *`）；就算你强转了，输出也是乱码。
- 保留原来的 `if (v.is_string) fatal_at(...)` 又加了新分支 → 那行永远走不到（因为新分支先判了 `is_string`），留着无害但会让人困惑。**本步的做法是先改结构、再改判断**，所以看起来是一次性换掉的。

#### 第 4 段：`N_UNARY` 加一行类型检查（白名单 W-2 之一）

```diff
         case N_UNARY: {
             Value a = eval(n->a, e);
+            if (a.is_string) fatal_at(n->tok, "一元运算符不能用于字符串");
             if (n->op == T_MINUS) return number(-a.value);
             if (n->op == T_PLUS)  return number(a.value);
```

**逐行解释**：

- 一元运算符有三个：`-x`、`+x`、`!x`。它们都作用在**一个**值上。
- `a.is_string` 检查放在**紧跟 `eval` 之后**，在读 `a.value` 之前。这样字符串不会走到 `number(-a.value)` 里被当成 0。
- `fatal_at(n->tok, ...)` 用的是 `N_UNARY` 节点的 token 位置。

**如果你写错了会怎样**：

- 漏掉这一行 → `-"hi"` 会输出 `0`（`-0.0`），不报错。**静默的错误**。
- 把它写在一元运算符的 `if` 之后 → `!` 和 `-` 走的是不同的分支，总有一个漏网。

**真实运行**：

```sh
$ cat /tmp/unary_str.ml
let a = -"x";
$ ./minilang /tmp/unary_str.ml
错误 [1:9]: 一元运算符不能用于字符串
```

#### 第 5 段：`N_BINARY` 加两行类型检查（白名单 W-2 之二）

```diff
             Value y = eval(n->b, e);
+            /* 字符串只能出现在 print 里，不能参与运算 */
+            if (x.is_string || y.is_string)
+                fatal_at(n->tok, "算术和比较运算只能用于数字");
             double a = x.value, b = y.value;
```

**逐行解释**：

- 检查在 `eval(n->b, e)` **之后**、`double a = x.value` **之前**。位置很关键：必须在把值"降级"成 `double` 之前拦下来。
- `x.is_string || y.is_string` —— **两边都要查**。`"a" + 1` 和 `1 + "a"` 都是非法的。
- 短路求值的那两行（`T_AND` / `T_OR`）在**上面**，所以 `"a" && 1` 会在算到 `y` 之前就短路返回……不对，仔细看：`x` 是 `"a"`，`x.value` 是 0，所以 `"a" && 1` 会走 `if (n->op == T_AND && x.value == 0) return number(0);` 直接返回 0，**根本不会报类型错误**。这是第 4 步短路求值留下的一个"漏洞"，但对教学版来说无所谓（`&&`/`||` 的结果本来就是 0/1）。

**如果你写错了会怎样**：

- 检查写在 `double a = x.value, b = y.value;` **之后** → 字符串已经被降级成 0 了，`x.is_string` 仍然是 1（`is_string` 字段没被覆盖），所以其实还能拦住；但如果哪天 `Value` 的布局变了就危险了。**检查一律放在降级之前。**
- 只检查 `y.is_string` 不检查 `x.is_string` → `"a" + 1` 会静默算出 1。

**真实运行**：

```sh
$ cat /tmp/bin_str.ml
let a = "x" + 1;
$ ./minilang /tmp/bin_str.ml
错误 [1:13]: 算术和比较运算只能用于数字
```

### 6.3 完整 diff

> **怎么用这段 diff**：推荐**照着敲**——每改一小段就编译一次，比一次性粘贴更容易记住。
> 如果想直接应用，把下面的内容存成 `step.patch`，然后在**上一步的目录**里执行：
>
> ```sh
> patch -p1 < ../009/step.patch
> ```
>
> 应用完和本目录的 `minilang.c` **逐字节一致**（可以 `diff` 验证）。

<details>
<summary>点开看完整 diff（真实 diff -u 输出）</summary>

```diff
--- a/minilang.c
+++ b/minilang.c
@@ -756,6 +756,30 @@
     fatal_at(t, "不能给未定义变量 '%s' 赋值", name);
 }
 
+/* 打印字符串模板：把 {变量名} 换成变量的值。
+   逐字符扫描，遇到 '{' 就找配对的 '}'。 */
+static void print_template(const char *s, Env *e, Token tok) {
+    while (*s) {
+        if (s[0] == '{') {
+            const char *end = strchr(s + 1, '}');
+            if (end) {
+                size_t len = (size_t)(end - (s + 1));
+                char name[MAX_NAME];
+                if (len == 0 || len >= sizeof(name))
+                    fatal_at(tok, "模板里的变量名无效");
+                memcpy(name, s + 1, len);
+                name[len] = '\0';
+                /* 模板里的变量必须已经定义过 */
+                double v = get_var(e, tok, name);
+                printf("%.15g", v);
+                s = end + 1;
+                continue;
+            }
+        }
+        putchar(*s++);
+    }
+}
+
 /* 一个值。MiniLang 里普通变量只能是数字，字符串只出现在 print 里，
    所以 Value 用一个标记位区分两者。
    is_string 和 string 这两个字段现在还没用上，是提前留好的。 */
@@ -799,23 +823,31 @@
         case N_NUM:
             return number(n->num);
 
-        case N_STR:
-            fatal_at(n->tok, "求值器还不支持字符串，字符串会在讲 print 模板时支持");
-            break;
+        case N_STR: {
+            /* 字符串字面量：求值结果是一个字符串值 */
+            Value v;
+            v.value = 0;
+            v.is_string = 1;
+            snprintf(v.string, sizeof(v.string), "%s", n->text);
+            return v;
+        }
 
         case N_VAR:
             /* 变量引用：沿着作用域链找 */
             return number(get_var(e, n->tok, n->text));
 
         case N_CALL: {
-            /* print 特殊处理：它要输出，而且下一步还要访问作用域查模板变量 */
+            /* print 特殊处理：它要输出，而且还要访问作用域查模板里的变量 */
             if (!strcmp(n->text, "print")) {
                 if (n->count != 1)
                     fatal_at(n->tok, "函数 'print' 参数数量错误：需要 1 个，实际 %d 个", n->count);
                 Value v = eval(n->items[0], e);
-                if (v.is_string)
-                    fatal_at(n->tok, "print 的参数必须是数字");
-                printf("%.15g", v.value);   /* 数字之间不会自动换行 */
+                if (v.is_string) {
+                    /* 字符串参数：当成模板渲染 */
+                    print_template(v.string, e, n->tok);
+                } else {
+                    printf("%.15g", v.value);   /* 数字之间不会自动换行 */
+                }
                 return number(0);
             }
 
@@ -839,6 +871,7 @@
 
         case N_UNARY: {
             Value a = eval(n->a, e);
+            if (a.is_string) fatal_at(n->tok, "一元运算符不能用于字符串");
             if (n->op == T_MINUS) return number(-a.value);
             if (n->op == T_PLUS)  return number(a.value);
             /* ! 的真值规则：0 是假，非 0 是真 */
@@ -854,6 +887,9 @@
             if (n->op == T_OR  && x.value != 0) return number(1);
 
             Value y = eval(n->b, e);
+            /* 字符串只能出现在 print 里，不能参与运算 */
+            if (x.is_string || y.is_string)
+                fatal_at(n->tok, "算术和比较运算只能用于数字");
             double a = x.value, b = y.value;
             switch (n->op) {
                 case T_PLUS:    return number(a + b);
```

</details>

> 顺便确认一下：diff 里**没有**文件头的变化。这是刻意设计的——`009/minilang.c` 和 `008/minilang.c` 的头注释一字不差，所以 diff 里只剩下真正的功能改动。

### 6.4 插入位置说明

如果你要自己在 `008/minilang.c` 上手改，五个位置分别在这里：

1. **`print_template()` 函数**：放在 `static void set_var(Env *e, Token t, const char *name, double v)` 函数的**结束大括号之后**，也就是 `Value` 结构体定义**之前**。为什么放这里？因为它同时要用到 `Env`（第 5 步定义的）、`get_var()`（第 5 步定义的）、`fatal_at()`（第 1 步定义的）——三个都在它上面，放前面才不会因为"先用了后定义"而报错。
2. **`case N_STR:`**：在 `static Value eval(Node *n, Env *e)` 的 `switch (n->type)` 里，在 `case N_NUM:` 的 `break;` 之后、`case N_VAR:` 之前。
3. **`print` 分支**：同一个 `switch` 里，在 `case N_CALL: {` 内部的 `Value v = eval(n->items[0], e);` 之后。
4. **`case N_UNARY:`**：在 `Value a = eval(n->a, e);` 这一行**紧接着的下一行**。
5. **`case N_BINARY:`**：在 `Value y = eval(n->b, e);` 这一行**紧接着的下一行**。

改完后编译：

```sh
cc -std=c11 -Wall -Wextra -O2 minilang.c -lm -o minilang
```

## 7. 测试

`./run.sh` 的真实输出：

```
▶ 用例（模式：run）
  ✅ case_01_template.ml          [run]
  ✅ case_02_escape.ml            [run]
  ✅ case_03_template_undefined.ml [run]
  ✅ case_04_string_in_arith.ml   [run]
  ✅ case_05_let_string.ml        [run]
  ✅ case_06_plain_string.ml      [run]
▶ demo.ml
  | x = 10, y = 32
  | x + y = 10

✅ 全部通过（7 项）
```

逐个用例说明：

**`tests/case_01_template.ml`** —— 模板替换：

```
let x = 10;
print("x = {x}\n");
```

- 期望 stdout：`x = 10` + 换行（`tests/case_01_template.stdout` 里是 7 个字节：`x = 10\n`）。
- 期望 stderr：空。期望退出码：0。
- 为什么：`print_template` 扫到 `{` → 找配对的 `}` → 变量名 `x` → `get_var` 找到 10 → `printf("%.15g", 10)` 输出 `10` → 继续扫到 `\n`（真的换行符）→ `putchar` 输出。

**`tests/case_02_escape.ml`** —— 转义：

```
print("a\tb\n");
```

- 期望 stdout：`a` + TAB + `b` + 换行（4 个字节）。
- 期望 stderr：空。期望退出码：0。
- 为什么：`\t` 在**词法分析阶段**就解码成了真的制表符，模板扫描时它只是一个普通字符，被 `putchar` 原样输出。

**`tests/case_03_template_undefined.ml`** —— 模板里的变量没定义：

```
print("{nope}\n");
```

- 期望 stderr（全文，一行）：

  ```
  错误 [1:1]: 未定义变量 'nope'
  ```

- 期望 stdout：空。期望退出码：1。
- 为什么报的是 `[1:1]` 而不是 `[1:8]`：`print_template` 传的是 `n->tok`（`print` 这个调用 token 的位置），**不是**字符串字面量的位置。语义上"模板里的变量没定义"是这次 `print` 调用的毛病。
- 为什么是"未定义变量"而不是"模板里的变量名无效"：名字 `nope` 是合法的标识符、长度也够，只是**没定义过**。`get_var()` 沿作用域链找不到，就报它。

**`tests/case_04_string_in_arith.ml`** —— 字符串参与算术：

```
let s = "hi";
let n = s + 1;
```

- 期望 stderr（全文，一行）：

  ```
  错误 [1:1]: 变量目前只支持数字类型
  ```

- 期望 stdout：空。期望退出码：1。
- 注意报的是**第 1 行**而不是第 2 行！因为第一行 `let s = "hi";` 就已经失败了——`exec` 的 `case N_LET:` 里的 `is_string` 检查先触发。**程序是从上往下执行的，第一个出错的语句就终止了。**
- 这个用例验证的是"字符串不能存进变量"，而不是"字符串不能参与运算"。想看后者的报错，看下面的 Q7。

**`tests/case_05_let_string.ml`** —— 直接把字符串存进变量：

```
let s = "hi";
```

- 期望 stderr（全文，一行）：

  ```
  错误 [1:1]: 变量目前只支持数字类型
  ```

- 期望 stdout：空。期望退出码：1。
- 为什么报在 `[1:1]`：`fatal_at(n->tok, ...)` 里的 `n` 是 `N_LET` 节点，它的 `tok` 是 `let` 关键字。

**`tests/case_06_plain_string.ml`** —— 完全没有变量的纯字符串：

```
print("没有变量\n");
```

- 期望 stdout：`没有变量` + 换行（13 字节 = 3 个汉字 ×3 字节 + 换行）。
- 期望 stderr：空。期望退出码：0。
- 为什么：`print_template` 一个 `{` 都没遇到，全程走 `putchar(*s++)` 那条路。**模板是可选的**——没有占位符的字符串就是普通字符串。

**`demo.ml` 冒烟测试** —— 输出两行（见第 3 节）。注意第二行输出的是 `x + y = 10` 而不是 `42`——**模板不做算术**。

## 8. 易错点与坑

### 坑 1：以为 `let s = "hi";` 能成功

- **错误现象**：写 `let s = "hi";` 之后满心欢喜地想 `print(s);`，结果得到 `错误 [1:1]: 变量目前只支持数字类型`。
- **原因**：MiniLang 的变量表只存 `double`。字符串是**字面量**，只能在 `print` 里当场用掉。
- **正确写法**：想打印就当场打印：`print("hi\n");`。想让一段文字重复出现，本步还做不到（没有字符串变量，也没有字符串拼接）。

### 坑 2：模板里写表达式

- **错误现象**：`print("{sqrt(81)}\n");` 报 `错误 [1:1]: 未定义变量 'sqrt(81)'`。
- **原因**：模板只支持 `{标识符}`。`strchr` 找到的 `}` 之前的内容被**整体**当成变量名——`sqrt(81)` 这个"名字"当然找不到。
- **正确写法**：先赋值再打印：

  ```
  let r = sqrt(81);
  print("sqrt(81) = {r}\n");
  ```

  这也是项目总览 README 里明确写下的限制："`print` 模板目前支持 `{变量名}`，例如 `{sum}`；不支持在模板内直接写 `{sqrt(81)}` 这样的表达式。可以先将函数结果赋给变量再输出。"

### 坑 3：以为 `print` 会自动换行

- **错误现象**：`print(42); print(43);` 输出 `4243`，一整行数字挤在一起。
- **原因**：`print` 从第 3 步起就不换行（源码里那行注释"数字之间不会自动换行"）。
- **正确写法**：字符串结尾加 `\n`：`print("42\n");`，或者干脆用 `print(42); print("\n");`。

### 坑 4：`Value` 的三个字段只初始化了两个

- **错误现象**：你在别处写了个 `Value v;` 然后只设了 `v.value` 和 `v.is_string`，没管 `v.string`。
- **原因**：`Value` 是**栈上的结构体**，不是全局变量，不会被自动清零。`v.string` 里是上一轮栈上的垃圾字节。
- **正确写法**：永远用工厂函数 `number()` 造数字值；造字符串值就照抄 `N_STR` 分支那样把三个字段都赋值。**或者干脆在声明处写 `Value v = {0};`**——C99 之后这能把所有字段清零，是最省心的写法。

### 坑 5：`snprintf` 会**静默截断**

- **错误现象**：字符串特别长（超过 4095 字节）时，输出莫名其妙地被砍掉一截，**而且不报错**。
- **原因**：`snprintf(dst, size, "%s", src)` 在空间不够时会**截断**并保证结尾有 `\0`，但**返回值**（应该是写入的完整长度）我们没检查。C 标准的 `snprintf` 和 Windows 上的 `_snprintf` 在这点上还不一样（后者不补 `\0`）。
- **正确写法**：本步的字符串最长是 `MAX_TEXT` = 4096，两边等长，理论上不会截断。但如果你以后把 `Value.string` 改小，记得检查 `snprintf` 的返回值：

  ```c
  int n = snprintf(v.string, sizeof(v.string), "%s", n->text);
  if (n < 0 || (size_t)n >= sizeof(v.string))
      fatal_at(n->tok, "字符串太长");
  ```

  另外，词法分析器里**已经**有一道长度检查（`if (n + 1 >= sizeof(b)) fatal_at(..., "字符串太长")`），所以运行期的截断其实很难触发。

### 坑 6：模板里 `{}` 或者超长的变量名

- **错误现象**：`print("a{}b");` 报 `错误 [1:1]: 模板里的变量名无效`。
- **原因**：`len == 0` 被判为非法。同样地，变量名长度 ≥ `MAX_NAME`（128）也报这个错。
- **正确写法**：模板里必须写一个合法的标识符。**注意这个错是在扫描到 `{` 的时候就报的**，而不是等到查变量——因为空名字连"查"都没法查。

### 坑 7：字符串里的 `{` 没有配对的 `}`

- **错误现象**：`print("a{b");` 输出 `a{b`，**不报错**。
- **原因**：`strchr` 返回 `NULL`，代码选择"当成普通字符输出"这个宽容策略。
- **正确写法**：这是设计上的取舍（宽容 vs 严格）。如果你觉得"漏写 `}` 应该报错"，那就得在 `if (end)` 后面补一个 `else fatal_at(...)`——但那样想打印一个真的花括号的用户就没辙了。**教学版选了宽容。**

### 坑 8：短路求值让类型检查漏网

- **错误现象**：`print("a" && 1);` 不会报"算术和比较运算只能用于数字"。
- **原因**：`N_BINARY` 里短路求值的两行在类型检查**之前**。`"a"` 的 `value` 是 0，所以 `x.value == 0` 成立，`&&` 直接返回 0，`y` 根本没被求值，类型检查也跳过了。
- **正确写法**：接受它。`&&` / `||` 的结果本来就是 0 或 1，用在字符串上虽然没报错但也不会得到有意义的结果。**如果你要严格的类型系统，就得把类型检查挪到短路判断之前。**

### 坑 9：`%` 在 `printf` 里不能用于 `double`

- **错误现象**：你想写 `print("{x}%\n");` 输出百分比，结果报错或者输出乱码。
- **原因**：这是 C 的经典坑——`printf` 的 `%` 后面跟 `f` 才是浮点数，`%` 单独出现是非法的格式说明符。
- **正确写法**：`print_template` 用的是 `putchar` 逐字节输出，**根本不经过 `printf` 的格式解析**，所以字符串里的 `%` 是安全的：`print("{x}%\n");` 会原样输出 `10%`。**只有你把 `%` 写进 `printf("...")` 的格式串里才会出事。**

## 9. 自测题

### 9.1 概念题

**Q1.** `let s = "hi";` 报"变量目前只支持数字类型"。如果 MiniLang 想支持字符串变量，最少要改哪几个地方？

**Q2.** `Value` 里已经有 `double value` 了，为什么还要额外加 `int is_string`？举一个"用 value 判断是不是字符串会出错"的例子。

**Q3.** 模板里写 `"{x + y}"` 会发生什么？为什么？

**Q4.** `print("a\tb\n");` 在 `tokens` 模式里显示成 `"a\tb\n"`，这个 `\t` 是源码里的两个字符，还是 token 里存的真实制表符？

**Q5.** 为什么本步必须修改 `N_UNARY` 和 `N_BINARY` 里已有的行？能不能不改？

<details>
<summary>参考答案</summary>

**A1.** 至少要改三处：(1) `Var` 结构体——把 `double value` 换成 `Value value`；(2) `env_push` / `find_var` / `set_var` / `get_var` 这一整套变量表操作；(3) `N_LET` / `N_ASSIGN` 里去掉"只支持数字"的检查。再加上"字符串能不能相加""字符串能不能当条件"等一连串语义问题。**这就是教学版把字符串限定为字面量的原因**——代价小，功能少，但每一步都讲得清楚。

**A2.** 因为 `0` 是一个完全合法的字符串（`print("")`）。如果用 `value == 0` 判断，`""` 会被当成数字 0。更根本的是 `Value` 里 `value` 和 `string` 两个字段**可能同时有内容**，必须有一个明确的"标签"告诉代码该读哪个。

**A3.** 报 `错误 [1:1]: 未定义变量 'x + y'`。因为模板扫描是纯文本级的：`strchr` 找到 `}`，把 `}` 之前的内容**整体**当成变量名（`x + y`），交给 `get_var` 查——查不到。

**A4.** token 里存的是**真实的制表符**。词法分析器在 `add_token` 之前就把 `\t` 解码成了 0x09。`tokens` 模式里看到的 `"a\tb\n"` 是 `print_escaped()` 把它**重新转义**成可读形式显示出来的。解码发生在词法阶段，显示转义发生在打印阶段，两件事分开。

**A5.** 不能不改。第 8 步的 `N_BINARY` 是 `double a = x.value, b = y.value;`，字符串的 `value` 是 0，不加检查的话 `"a" + 1` 会**静默算出 1**——不报错，程序"能跑"，但结果是错的。这比报错糟糕得多。所以必须加检查，这就是白名单 W-2 存在的理由。

</details>

### 9.2 预测题（"改一行会发生什么"）

**Q6.** 把 `N_STR` 分支里的 `v.is_string = 1;` 删掉，`tests/case_05_let_string.ml` 的输出会变吗？`tests/case_01_template.ml` 呢？

**Q7.** 把 `N_BINARY` 里的类型检查那两行删掉，`print("x" + 1);` 会输出什么？

**Q8.** 把 `print_template` 里的 `continue;` 删掉，`print("{x}")`（`x = 10`）会输出什么？

**Q9.** 把 `print_template` 里的 `name[len] = '\0';` 删掉，会发生什么？

**Q10.** 把 `print_template` 里的 `strchr(s + 1, '}')` 改成 `strchr(s, '}')`，`print("{x}")` 会输出什么？

<details>
<summary>参考答案</summary>

**A6.** `case_05_let_string.ml` **会变**：不再报错，`let s = "hi";` 会成功定义一个值为 0 的变量，程序正常退出（退出码 0，stdout 空）。`case_01_template.ml` **也会变**：`print("x = {x}\n")` 会输出 `x = 0` 而不是 `x = 10`——因为 `is_string` 是 0，走了数字分支，`printf("%.15g", v.value)` 打印的是字符串的 `value`（0）。

**A7.** 输出 `1`，**不报错**。`"x"` 的 `value` 是 0，`0 + 1 = 1`。这正是 W-2 要防的事：静默的错误结果。

**A8.** 输出 `10}`。少了 `continue`，代码会接着执行 `putchar(*s++)`，而此时 `s` 还指着 `'}'`（因为 `s = end + 1` 是在 `continue` **之前**执行的……严格说：`s = end + 1` 先把 `s` 移到 `}` 之后，然后没有 `continue` 就执行 `putchar(*s++)`，输出的是 `}` 之后的那个字符）。**准确地说**：你把 `s = end + 1` 和 `continue` 一起留着但删掉 `continue`，那么 `putchar(*s++)` 输出的是 `}` **后面**的字符（即 `}` 后一字节）然后 `s` 再前进一位——结果是**漏掉了 `}` 本身，但多输出了 `}` 后面的那个字符**。要得到 `10}`，得把 `s = end + 1;` 也删掉，只留 `continue` 前的那行。

**A9.** `name` 数组里 `name[len]` 之后的内容是**未初始化的栈垃圾**。`get_var` 里的 `strcmp(e->vars[i].name, name)` 会拿垃圾当字符串比较——可能"碰巧"匹配上某个变量，可能匹配不上报"未定义变量 'x + 随机字符'"，最坏情况是 `strcmp` 越界读、段错误。**这是一个典型的未定义行为（undefined behavior）：同一份代码换个编译器就可能表现不同。**

**A10.** 报 `错误 [1:1]: 未定义变量 '{x'`。因为 `strchr(s, '}')` 从 `{` 自己开始找，**最先找到的就是 `{` 后面那个 `}`**——等等，`strchr(s, '}')` 找的是**字符 `}`**，不是 `{`。从 `s`（指向 `{`）开始找 `}`，找到的就是配对的 `}`，位置和原来一样……**所以输出不变，还是 `10`。** 再想想：`strchr(s, '}')` 找的是 `}` 字符，`s` 指向 `{`，往后第一个 `}` 就是配对的那个，和 `strchr(s+1, '}')` 结果**完全相同**。真正会出问题的是把 `strchr(s + 1, '}')` 改成 `strchr(s + 2, '}')`——那会跳过变量名的第一个字符。

</details>

### 9.3 动手题（改代码 + 跑 tests 验证）

**Q11.** 写一个 `.ml`，让 `print` 输出"变量未定义"的错误，但错误位置指向**字符串字面量**而不是 `print`。做法：改 `print_template` 的第三个参数（`tok`），从 `n->tok` 改成字符串字面量自己的 token。看清楚 `eval` 里 `N_CALL` 节点有没有保存参数节点的 token。

**Q12.** 给 MiniLang 加一个"字符串长度"的内置函数 `len(s)`——不需要，先想清楚：如果要加，`Builtin` 表是"数字进、数字出"的，`len` 要吃字符串怎么办？（提示：看 `BuiltinFn` 的签名 `double (*)(double *args, Token tok)`。）

**Q13.** 改进 `print_template`：让 `{变量名}` 里如果写的是**未定义**的变量，错误信息里能显示是哪一段模板出的问题（比如 `未定义变量 'nope'（在模板 "xxx {nope} yyy" 中）`）。

**Q14.**（进阶）支持模板里的 `{{` 转义：`print("{{x}}")` 应该输出字面的 `{x}`。提示：在扫描到 `{` 时先看看下一个字符是不是 `{`。

<details>
<summary>参考答案</summary>

**A11.** `eval` 的 `N_CALL` 分支里，`n->items[0]` 是参数节点，它的 `tok` 字段就是字符串字面量的 token。想让错误指向字符串，需要在 `N_CALL` 分支里把 `n->items[0]->tok` 传进去：

```c
print_template(v.string, e, n->items[0]->tok);
```

改完之后 `print("{nope}\n");` 报 `错误 [1:7]: 未定义变量 'nope'`（`1:7` 是 `{` 的位置，因为 `{` 在字符串 token 的 text 里，字符串 token 的 `col` 是**左引号**的位置……注意：字符串 token 的 `col` 是 `"` 的列号，`{` 在引号里面第 2 个字符，所以真正的 `{` 应该是第 8 列。词法分析器给整个字符串只记了一个 `col`，也就是引号的位置 `1:7`。）

**A12.** `BuiltinFn` 的签名是 `double (*)(double *args, Token tok)`——参数是 `double` 数组，**字符串进不来**。所以 `len` 不能放进 `Builtin` 表。有两条路：(1) 把 `BuiltinFn` 改成吃 `Value *args`（要改表里所有 8 个内置函数，代价大）；(2) 像 `print` 那样，在 `eval` 的 `N_CALL` 里**单独特判**。教学版选了 (2) 那种思路（`print` 就是这么做的）。**这就是"print 不在 Builtin 表里"的深层原因。**

**A13.** 一种做法是给 `print_template` 加第四个参数 `const char *whole`（整条模板），在报错时带上：

```c
fatal_at(tok, "未定义变量 '%s'（模板：\"%s\"）", name, whole);
```

但 `get_var()` 是第 5 步写死的通用函数，它不知道模板的事。更干净的做法是在 `print_template` 里**自己查变量**（用 `find_var` 沿链找），而不是调 `get_var`，这样就能自定义错误信息。**代价是重复实现了 `get_var` 的链式查找逻辑**——教学版选择复用，接受"错误信息不够具体"。

**A14.** 核心思路：在 `if (s[0] == '{')` 成立之后，先判断 `s[1] == '{'`：

```c
if (s[0] == '{') {
    if (s[1] == '{') { putchar('{'); s += 2; continue; }   /* 转义输出 {{ */
    ...
}
```

但要注意 `{{x}}` 的**配对**：输出 `{` 之后，`s` 指向 `x`，然后 `x` 会被当成普通字符输出，接着 `}` 也会被当普通字符输出——结果 `{x}`，正好是你要的。所以这个改法能work。**但它改变了"没有配对 `}` 的 `{` 原样输出"这条规则**，需要一起想清楚。

</details>

## 10. 本步的设计取舍

**为什么字符串不能存进变量？**

这是本步最大的一个取舍，代价是"程序里没法保存一段文字"。换来的是：`Env` 的变量表还是 `double`，`let` / 赋值 / 作用域这一整套（第 5 步）**一个字都没改**。

如果支持字符串变量，至少要动：`Var` 结构体、`env_push`、`find_var`、`set_var`、`get_var`、`N_LET`、`N_ASSIGN`，还要回答"两个字符串相加是什么意思""字符串能不能当 `if` 的条件""函数能不能返回字符串"——这一串问题会把教学版变成一个真正的小语言，工作量是现在的三倍以上。**值不值？对一个"教你写解释器"的教程来说，不值。**

**为什么 `print` 是特例而不是普通函数？**

`print` 要做三件别的事：(1) 往 stdout 写东西；(2) 访问当前作用域来查模板里的变量；(3) 接受字符串参数。普通函数是"数字进、数字出"的纯函数（`Builtin` 表），第 8 步的注释已经把这条线画清楚了。**`print` 的需求和它们正交，所以必须特殊处理。**

代价是：`print` 不能被赋值给变量（`let f = print;` 不支持）、不能作为值传递、不能递归调用自己。第 13 步模块化重构时，`print` 会是唯一一个"住在解释器模块里而不是内置函数表里"的函数。

**为什么模板只支持 `{标识符}`，不支持 `{表达式}`？**

支持表达式意味着模板里要跑一个迷你求值器：先切出 `}` 之前的文本，再调 `expression()` 解析它，再求值。技术上不难（第 2 步的解析器现成），但会带来一堆麻烦：

- 模板里的表达式出错时，行列号怎么算？（模板在源码里的位置和它内部的子位置不是一回事）
- 模板里的表达式能不能有副作用？（`{f()}` 里的 `f` 能不能打印东西？）
- `{sqrt(81)}` 里 `sqrt` 找不到怎么办？

**教学版选了"只支持标识符"**：模板渲染时唯一可能出的错就是"变量没定义"，而且这个错有明确的行列号（`print` 调用所在的位置）。**牺牲功能换清晰。**

**代价：模板里不能算术。** `print("{x + y}\n")` 不行，必须先 `let s = x + y;` 再 `print("{s}\n")`。

**为什么 `print` 不返回有用的值？**

`print` 恒定返回 `number(0)`。这不是疏忽——`print` 是**副作用**函数，返回值本来就没意义。留着 0 是因为 `eval` 必须返回一个 `Value`。

**还没做的：**

- 字符串拼接（`"a" + "b"`）。
- 字符串比较。
- 转义 `{` / `}`（现在只能用 `{{` 硬绕，但那还没实现）。
- 字符串作为 `if` 的条件（会报"if 的条件必须是数字或比较表达式"）。
- `%d` / `%s` 之类的**格式化占位符**——模板是"变量替换"，不是"格式串"。`print("{x}%%\n")` 会输出 `10%%`，两个百分号都在，因为 `%` 在 `putchar` 路径上没有特殊含义。

## 11. 下一步预告

到这里，MiniLang 已经能算、能分支、能循环、能用内置函数、能打印带变量的字符串了。但程序还是"从上到下顺序执行一遍"——写完就结束，不能复用。

下一节我们加入**用户自定义函数**：`fn add(a, b) { print(a + b); }` 和 `add(3, 4);`。你会看到三个新东西：一个新语法（`fn` 声明）、一张新的运行时表（`Func` 表）、一个必须靠"先注册再执行"才能解决的问题（**函数之间互相调用**，这是递归的前提）。还会看到 `dupstr` 这个小函数背后藏着的 C 语言指针生命周期陷阱，以及一个刻意留到下一步的"半成品"——**本步的函数没有 `return`，函数的值恒为 0**。

下一节：[第 10 步 · 用户函数：定义与调用](../010/README.md)
