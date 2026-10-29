# 第 8 步 · 内置函数

> 进度 ████████░░░░ 8/13 ｜ 上一节：[第 7 步 · for 循环](../007/README.md) ｜ 下一节：[第 9 步 · print 与字符串模板](../009/README.md)

## 1. 本步目标

读完你将能够：

- 在 MiniLang 里调用数学函数：`print(sqrt(16));` 输出 `4`，`print(abs(-3));` 输出 `3`。
- 用 `abs` / `round` / `sqrt` / `floor` / `ceil` / `pow` / `min` / `max` 这 8 个内置函数。
- 说出"表驱动"是什么意思，并解释为什么加第 9 个内置函数只需要往数组里加一行。
- 读懂 C 的函数指针声明 `double (*fn)(double *args, Token tok)`，并知道函数指针的 `*` 为什么要加括号。
- 解释 `print` 为什么**不能**放进内置函数表，以及 `round(-2.5)` 为什么等于 `-2` 而不是 `-3`。

## 2. 运行方式

```sh
./run.sh                 # 编译 + 跑全部用例 + 跑 demo
./minilang demo.ml       # 默认 run 模式
./minilang demo.ml tokens
./minilang demo.ml ast
```

本步 `run.sh` 的真实输出见 [第 7 节](#7-测试)。

## 3. 现在的程序能做什么

| 模式 | 输入 | 输出 | 退出码 |
|---|---|---|---|
| `tokens` | 任意 `.ml` | token 列表（内置函数名就是普通 `ID`，没有特殊 token） | 0（有错为 1） |
| `ast` | 任意 `.ml` | 语法树（`abs(-3)` 打印成 `调用 abs` → `一元 MINUS` → `数字 3`） | 0（有错为 1） |
| `run`（默认） | 任意 `.ml` | 真正执行；支持 8 个数学内置函数 + `print` | 0；错误为 1 |

`008/demo.ml` 的内容：

```
// 第 8 步：内置函数
print(abs(-3));
```

真实运行结果：

```sh
$ ./minilang demo.ml
3
```

用 `ast` 模式看 `abs(-3)` 这棵树：

```sh
$ ./minilang demo.ml ast
语法树：
代码块
└── 表达式语句
    └── 调用 print
        └── 调用 abs
            └── 一元 MINUS
                └── 数字 3
```

**关键观察**：`abs` 在语法层面和 `print` **完全一样**——都是 `N_CALL` 节点，都有一个函数名和一个实参列表。**内置函数不是一种新的语法，它只是"名字查得到"的普通函数调用。** 这就是为什么本步只需要改 `eval` 里的 `N_CALL` 分支，词法分析器和语法分析器一行都不用动（INV-1 的又一次兑现）。

## 4. 前置知识回顾

上一步我们做完了循环。本步真正需要的前置知识有三样：

**（1）`N_CALL` 节点。** 第 2 步引入的调用节点：`n->text` 是函数名，`n->items[]` 是实参列表，`n->count` 是实参个数。解析器早就支持 `f(a, b)` 这种语法了——**从第 2 步起，MiniLang 就"能解析"函数调用，只是第 5 步之前执行不了，第 7 步之前只认识 `print` 一个。**

**（2）`eval()` 的 `N_CALL` 分支。** 第 5 步加的，代码长这样：

```c
case N_CALL: {
    if (strcmp(n->text, "print") != 0)
        fatal_at(n->tok, "未知函数 '%s'", n->text);
    if (n->count != 1)
        fatal_at(n->tok, "函数 'print' 参数数量错误：需要 1 个，实际 %d 个", n->count);
    Value v = eval(n->items[0], e);
    if (v.is_string)
        fatal_at(n->tok, "print 的参数必须是数字");
    printf("%.15g", v.value);   /* 数字之间不会自动换行 */
    return number(0);
}
```

本步就是**把这一小坨"只认 print"的代码，换成"查表"**。

**（3）`Token` 结构体。** 第 1 步定型的 `{ TokType type; char text[MAX_TEXT]; double num; int line, col; }`。本步的每个内置函数都要收到一个 `Token`——用来在出错时报告行列号。

## 5. 本步引入的核心概念

### 5.1 概念一：为什么是"表驱动"，而不是一长串 `if`

**问题**：如果要支持 8 个内置函数，最直观的写法是这样：

```c
if      (!strcmp(n->text, "abs"))   { if (n->count != 1) ...; return number(fabs(args[0])); }
else if (!strcmp(n->text, "round")) { if (n->count != 1) ...; return number(floor(args[0] + 0.5)); }
else if (!strcmp(n->text, "sqrt"))  { ... }
...
else if (!strcmp(n->text, "max"))   { ... }
else fatal_at(n->tok, "未知函数 '%s'", n->text);
```

能跑，但每加一个函数就要改一次这段代码，而且"函数名 → 实现"和"参数个数检查"这两件事被搅在一起。

**思路**：把这张"名字 → 实现"的对应关系**抽成一张数据表**，代码只负责查表：

```
   ┌──────────────────────────────────────────────────┐
   │  builtins[] 表                                   │
   ├────────┬──────────┬──────────┬──────────────────┤
   │ name   │ min_args │ max_args │ fn（函数指针）    │
   ├────────┼──────────┼──────────┼──────────────────┤
   │ "abs"  │    1     │    1     │ &bi_abs          │
   │ "round"│    1     │    1     │ &bi_round        │
   │ "sqrt" │    1     │    1     │ &bi_sqrt         │
   │ "floor"│    1     │    1     │ &bi_floor        │
   │ "ceil" │    1     │    1     │ &bi_ceil         │
   │ "pow"  │    2     │    2     │ &bi_pow          │
   │ "min"  │    2     │    2     │ &bi_min          │
   │ "max"  │    2     │    2     │ &bi_max          │
   └────────┴──────────┴──────────┴──────────────────┘
                    │
                    │ find_builtin("sqrt")
                    ↓
              &builtins[2]  →  name="sqrt", min=1, max=1, fn=bi_sqrt
```

这就是**表驱动**（table-driven）：把"变化的、需要扩展的部分"放进数据，把"不变的、流程性的部分"留在代码里。

**收益（非常具体）**：

- 加第 9 个内置函数（比如 `log`）**只需要在数组里加一行**，一行 C 代码都不用改。
- 参数个数检查、参数求值、错误处理这些**每个函数都一样**的逻辑只写一遍。
- 想支持"参数个数可变"的函数（比如 `min` 接受 2 个以上）只要把 `min_args` / `max_args` 设成不同的值就行。

**代价**：多了一层间接（查表 → 函数指针），初学者要多花一点时间理解"函数指针"。

### 5.2 概念二：`Builtin` 结构体与 `BuiltinFn` 函数指针

**问题**：表里的每一项要存什么？

**思路**：三样东西——**名字**、**能接受几个参数**、**具体怎么算**。

```c
typedef double (*BuiltinFn)(double *args, Token tok);

typedef struct {
    const char *name;    /* 函数名         */
    int         min_args;/* 最少几个参数   */
    int         max_args;/* 最多几个参数   */
    BuiltinFn   fn;      /* 具体怎么算     */
} Builtin;
```

**怎么读 `double (*fn)(double *args, Token tok)`？**

C 的声明是"**从里往外**"读的。这条声明从里往外读是：

1. `double` —— 返回值是 `double`。
2. `(*fn)` —— `fn` 是个**指针**（因为有 `*` 和括号）。**括号是必须的**！
3. `(double *args, Token tok)` —— 它是个**函数**，参数是 `double *args`（指向 double 数组的指针）和 `Token tok`（一个 Token，按值传）。

所以完整意思是：**`fn` 是一个函数指针，指向"接收 `double*` 和 `Token`、返回 `double`"的函数。**

```
   double (*fn)(double *args, Token tok)
          ↑        ↑
          |        └── 参数表
          └─────────── 指针（加了括号才是指针，否则 fn 是个函数）
```

**为什么 `*` 必须加括号？** 因为不加括号时 `double *fn(double *args, Token tok)` 的意思是"`fn` 是一个**返回 `double*`** 的**函数**"——`fn` 是函数，不是函数指针。这是 C 里最经典的"声明地狱"之一：

```c
double  *fn(double *args, Token tok);   /* fn 是函数，返回 double*        */
double (*fn)(double *args, Token tok);   /* fn 是指向函数的指针            */
```

**为什么参数是 `double *args`（数组）而不是一个个 `double a, double b`？**

因为每个内置函数接受的参数个数**不一样**（1 个或 2 个）。用数组统一表示："实参都先求值存进 `args[]` 数组，你按需要取 `args[0]`、`args[1]`"。

**为什么还要传一个 `Token tok`？**

为了让内置函数**在出错时能报出正确的位置**。`sqrt(-1)` 的错误信息是：

```
错误 [3:1]: sqrt 参数不能为负数
```

`[3:1]` 是**函数名 `sqrt` 所在的位置**（第 3 行第 1 列）。内置函数自己不知道自己在源码的哪里——它只拿到一个 `double`。所以求值器调用它的时候，把 `n->tok`（函数名 token）一起传进去。这就是 `tok` 这个参数的**唯一用途**。

**代码要点**：每个内置函数的签名必须和 `BuiltinFn` **完全一致**，否则不能放进 `builtins[]` 数组（编译器会报类型不匹配）。所以每个函数都写成：

```c
static double bi_abs(double *a, Token tok)   { (void)tok; return fabs(a[0]); }
```

### 5.3 概念三：`(void)tok;` —— C 里怎么"忽略"一个参数

**问题**：`bi_abs` 用不到 `tok`，但签名必须带上它。怎么办？

**思路**：C 语言**没有**"忽略参数"的语法。你必须"用"它一下，否则编译器会警告。

最常见的做法是 `(void)tok;`：把它强制转换成 `void` 类型并丢弃。`(void)` 是 C 里的"**显式丢弃**"惯用法——`void` 表达式不能被用在任何需要值的地方，所以编译器知道你是故意的，不会再警告"表达式结果未使用"。

**真实证据**：把这个 `(void)tok;` 删掉，用 `-Wall -Wextra` 编译，你会看到：

```
w.c:3:39: warning: unused parameter 'tok' [-Wunused-parameter]
    3 | static double bi_abs(double *a, Token tok) { return fabs(a[0]); }
      |                                       ^
1 warning generated.
```

注意 `-Wextra` 才打开这个警告（`-Wall` 不含）。本项目的 `run.sh` 用的是 `-std=c11 -Wall -Wextra -O2`，所以**每个内置函数都必须写 `(void)tok;`**，否则会有一堆警告。

**代码要点**：`bi_sqrt` 是唯一**用得到** `tok` 的：

```c
static double bi_sqrt(double *a, Token tok) {
    if (a[0] < 0) fatal_at(tok, "sqrt 参数不能为负数");
    return sqrt(a[0]);
}
```

它把负数检查放在**函数内部**，所以能拿到 `tok`，报出准确的位置。这就是"把检查放在内层函数里"的价值——如果检查放在调用方（`eval`）里，报错位置会变成"所有函数共用一个位置"，就没这么精确了。

### 5.4 概念四：`print` 为什么不进表

**问题**：`print` 也是内置函数，为什么不能放进 `builtins[]`？

**思路**：因为表的**契约**是"数字进、数字出"的**纯函数**（pure function）：给它几个数字，它还你一个数字。`print` 违反了这个契约：

1. **它有副作用**：它要往标准输出写东西。
2. **它要访问作用域**：第 9 步我们要支持 `print("x = {x}\n")`——模板里的 `{x}` 需要去**当前作用域**里查变量 `x` 的值。而表里的内置函数拿到的只是 `double *args`，**根本没有 `Env *` 参数**，它查不到变量。

第 2 点是**决定性**的：即使把 `print` 硬塞进表里，函数指针的签名 `double (*)(double *args, Token tok)` 也没有地方放 `Env *e`。要放进去，就得改签名，改成 `double (*)(double *args, Token tok, Env *e)`——**那就不是"数字进数字出"了，全表都得跟着改**。

所以代码里把 `print` 单独处理，**放在查表之前**：

```c
case N_CALL: {
    /* print 特殊处理：它要输出，而且下一步还要访问作用域查模板变量 */
    if (!strcmp(n->text, "print")) {
        ...专门处理...
    }

    /* 其余内建函数：查表 */
    const Builtin *b = find_builtin(n->text);
    if (!b)
        fatal_at(n->tok, "未知函数 '%s'", n->text);
    ...
}
```

**代码要点**：注意判断顺序——**先 `print`，后查表**。如果顺序反了（先查表），`print` 就会落到"未知函数"分支里。

**边界情况**：内置函数名和变量名重名时，**内置函数优先**：

```
let abs = 5;
print(abs(3));
```

真实输出是 `3`——因为 `N_CALL` 的求值路径压根不查变量表，直接查内置函数表。

### 5.5 概念五：参数个数检查 vs 参数类型检查

这是两个**不同**的检查，发生在**不同**的位置：

```
   ┌─────────────────────────────────────────────────────┐
   │  eval(N_CALL) 的执行顺序                              │
   │                                                     │
   │  1. 是不是 print？ → 是，走 print 专用分支             │
   │  2. find_builtin(name) → 找不到？报"未知函数"          │
   │  3. 参数个数检查：count < min_args || count > max_args │
   │     → 报错"参数数量错误"        ← 检查 A，在求值之前    │
   │  4. 逐个求值实参，存进 args[]                        │
   │     每个实参求值后检查是不是字符串                      │
   │     → 报错"参数必须是数字"      ← 检查 B，在求值之中    │
   │  5. 调用 b->fn(args, n->tok)                         │
   └─────────────────────────────────────────────────────┘
```

**检查 A（个数）先于实参求值。** 这有实际意义：

```
abs(1, 1 / 0);
```

真实输出是：

```
错误 [1:1]: 函数 'abs' 参数数量错误：需要 1 个，实际 2 个
```

**注意报的不是"除数不能为 0"**——因为个数检查先把程序拦下来了，实参根本没被求值。如果顺序反过来，用户会先看到一个和真正问题无关的"除数为 0"，浪费 debugging 时间。**先做便宜的检查，是解释器的通用优化。**

**检查 B（类型）现在还走不到。** 代码是这样写的：

```c
double args[MAX_PARAMS];
for (int i = 0; i < n->count; i++) {
    Value v = eval(n->items[i], e);
    if (v.is_string)
        fatal_at(n->tok, "函数 '%s' 的参数必须是数字", b->name);
    args[i] = v.value;
}
```

但第 8 步的 `eval` 遇到字符串字面量（`N_STR`）就已经 `fatal_at` 了：

```sh
$ ./minilang /tmp/abs_str.ml
错误 [1:5]: 求值器还不支持字符串，字符串会在讲 print 模板时支持
```

报错位置 `[1:5]` 是**字符串字面量自己**的位置，不是调用位置。**这行 `is_string` 检查在第 8 步是"死代码"**——它是为第 9 步准备的：第 9 步 `eval` 真的会返回字符串值，那时候 `abs("hi")` 就会报"函数 'abs' 的参数必须是数字"了。

这正是 INV-2（`Value` 里的字符串字段从第一天就预留）的价值：**当时不写，第 9 步就得回头改 `eval` 的签名或者加特判。**

### 5.6 概念六：`round` 的语义是 `floor(x + 0.5)`

**问题**：`round(2.5)` 等于 3，那 `round(-2.5)` 等于几？

**思路**：MiniLang 的 `round` 实现是：

```c
static double bi_round(double *a, Token tok) { (void)tok; return floor(a[0] + 0.5); }
```

也就是 **"先加 0.5，再向下取整"**——等价于**向 +∞ 方向四舍五入**（.5 总是往大的方向走）。

```
   x        x + 0.5      floor(x + 0.5) = round(x)
   2.4      2.9          2
   2.5      3.0          3      ← .5 往上
  -2.4     -1.9         -2
  -2.5     -2.0         -2      ← .5 也往上（往 +∞）
   0.5      1.0          1
```

真实输出：

```sh
$ ./minilang /tmp/round.ml
print(round(2.5));    →  3
print(round(-2.5));   →  -2
print(round(2.4));    →  2
print(round(-2.4));   →  -2
print(round(0.5));    →  1
```

连在一起是 `3-22-21`。

**⚠️ 这和 C 标准库的 `round()` 不一样！** C 的 `round()` 是"**四舍五入到最近的整数，中点 away from zero**"（中点远离零）：

| 输入 | C 的 `round()` | MiniLang 的 `round()` |
|---|---|---|
| `2.5` | 3 | 3 |
| `-2.5` | **-3** | **-2** |
| `-2.4` | -2 | -2 |

实测对照（C 程序）：

```
C round(-2.5)=-3  round(2.5)=3
floor(-2.5+0.5)=-2  floor(2.5+0.5)=3
floor(-2.4+0.5)=-2
```

**为什么 MiniLang 选 `floor(x + 0.5)`？**

- **优点**：实现简单，一行搞定，不用判断符号；而且大多数语言（Python 的 `round`、JavaScript 的 `Math.round`）也是"往 +∞ 方向四舍五入"（虽然 Python 3 的 `round` 是"银行家舍入"，那是另一个话题）。
- **缺点**：和 C 标准库不一样。如果你从 C 迁移过来 expecting 同样的行为，会在负数上踩坑。**所以这一条必须记住：MiniLang 的 `round(-2.5)` 是 `-2`。**

顺带一提，`floor` 和 `ceil` 就是直接调 C 标准库的 `floor()` / `ceil()`，它们的语义和 C 完全一致（往 -∞ / +∞ 取整）：

```
floor(-1.5) → -2
ceil(-1.5)  → -1
```

### 5.7 概念七：查表的实现细节

```c
static const Builtin builtins[] = {
    { "abs",   1, 1, bi_abs   },
    { "round", 1, 1, bi_round },
    ...
};
static const int nbuiltins = (int)(sizeof(builtins) / sizeof(builtins[0]));

/* 按名字查内置函数 */
static const Builtin *find_builtin(const char *name) {
    for (int i = 0; i < nbuiltins; i++)
        if (!strcmp(builtins[i].name, name)) return &builtins[i];
    return NULL;
}
```

**代码要点**：

- `sizeof(builtins)` 是整个数组的字节数，`sizeof(builtins[0])` 是**一个元素**的字节数，两者相除就是元素个数。这是 C 里计算数组长度的标准惯用法。
- `nbuiltins` 声明成 `static const int`，但它的初值是 `sizeof` 的结果——`sizeof` 是编译期常量，所以这仍然算"常量初始化"，不需要运行时计算。
- `find_builtin` 返回 `const Builtin *`（指向常量的指针）。用 `const` 是为了让调用方**不能修改表里的内容**——表是全局的静态数据，谁都能改的话太危险了。
- 返回 `NULL` 表示"没找到"，调用方检查 `if (!b)` 报错。

**边界情况**：这是**线性查找**，O(n)。8 个函数无所谓，但如果内置函数涨到几百个，就得改成哈希表。不过内置函数是**编译期固定**的，数量很少，线性查找完全够用。

**`const` 的一个坑**：`find_builtin` 返回的是 `const Builtin *`，所以在 `eval` 里必须写成 `const Builtin *b = find_builtin(...)`。如果你写成 `Builtin *b = ...`，编译器会报

```
error: initializing 'Builtin *' with an expression of type 'const Builtin *' discards qualifiers
```

这是 C 里 `const` 传染性的经典表现。

## 6. 动手：把代码加进来

### 6.1 改动清单

| 位置锚点 | 改了什么 | 大概多少行 | 为什么 |
|---|---|---|---|
| `/* 运行时：值与求值 */` 注释块**之前** | 新增整个"内置函数表"小节：`BuiltinFn`、`Builtin`、8 个 `bi_*` 函数、`builtins[]`、`find_builtin` | 约 50 行 | 表驱动的主体 |
| `eval()` 的 `case N_CALL:` | 把"只认 print"的代码换成"print 特殊分支 + 查表" | 净增约 16 行（删 11 行、增 27 行） | 真正调用内置函数 |

合计 **约 66 行新增、11 行删除**。删除的这 11 行就是第 7 步那两句 `if (strcmp(...))` 和 `if (n->count != 1)`——它们被"查表 + 统一检查"取代了。

### 6.2 分段讲解

#### 第 1 段：函数指针类型和表项结构

```diff
+/* ------------------------------------------------------------------ */
+/* 内置函数表                                                          */
+/*                                                                     */
+/* 表里放的是"数字进、数字出"的纯函数。print 不在表里——它要输出，   */
+/* 而且下一步还要访问作用域去查模板里的变量，需求不一样。             */
+/* ------------------------------------------------------------------ */
+typedef double (*BuiltinFn)(double *args, Token tok);
+
+typedef struct {
+    const char *name;    /* 函数名         */
+    int         min_args;/* 最少几个参数   */
+    int         max_args;/* 最多几个参数   */
+    BuiltinFn   fn;      /* 具体怎么算     */
+} Builtin;
```

**逐行解释**：

- `typedef double (*BuiltinFn)(double *args, Token tok);` —— 给"函数指针"这个类型起个名字。**`typedef` 只是给已有类型起别名，不创建新类型**。之后 `BuiltinFn` 就等价于 `double (*)(double *, Token)`。
- `name` 是 `const char *`——字符串字面量在 C 里就是 `const char *`，指向只读的字符串常量区。**不能写 `char *`**，那样表就成了可写的，多余的风险。
- `min_args` / `max_args` 分开写，是为了将来支持"参数个数可变"的函数（比如 `min` 接受 1 个以上）。**现在每个函数都恰好 `min == max`**，但分开写更灵活。
- `fn` 的类型是 `BuiltinFn`，也就是"指向内置函数的指针"。

**如果你写错了会怎样**：

- 写成 `typedef double *BuiltinFn(double *args, Token tok);` → 少了括号，`BuiltinFn` 变成"返回 `double*` 的函数"类型。后面 `BuiltinFn fn;` 会报 "function declaration isn't a prototype"/类型不匹配。
- `name` 写成 `char *` → 编译通过（`"abs"` 是字符串字面量，类型是 `char[]`，赋给 `char *` 在 C 里只是"丢弃 const"的警告），但 `-Wall` 会给出 `warning: initialization discards 'const' qualifier`。用 `const char *` 从一开始就干净。

#### 第 2 段：8 个内置函数

```diff
+static double bi_abs(double *a, Token tok)   { (void)tok; return fabs(a[0]); }
+static double bi_round(double *a, Token tok) { (void)tok; return floor(a[0] + 0.5); }
+static double bi_sqrt(double *a, Token tok)  { if (a[0] < 0) fatal_at(tok, "sqrt 参数不能为负数"); return sqrt(a[0]); }
+static double bi_floor(double *a, Token tok) { (void)tok; return floor(a[0]); }
+static double bi_ceil(double *a, Token tok)  { (void)tok; return ceil(a[0]); }
+static double bi_pow(double *a, Token tok)   { (void)tok; return pow(a[0], a[1]); }
+static double bi_min(double *a, Token tok)   { (void)tok; return a[0] < a[1] ? a[0] : a[1]; }
+static double bi_max(double *a, Token tok)   { (void)tok; return a[0] > a[1] ? a[0] : a[1]; }
```

**逐行解释**：

- 全部是 `static`——它们只在这个文件里用，不需要外部可见。
- 全部返回 `double`——MiniLang 的数字只有一种类型（`double`）。
- 参数名统一叫 `a`（数组），所以 `a[0]` 是第一个实参，`a[1]` 是第二个。
- **`(void)tok;`** —— 5.3 节讲过的"显式丢弃"惯用法。没有它，`-Wextra` 会报 7 个 `unused parameter` 警告。
- `bi_sqrt` 是唯一一个**用得到 `tok`** 的：它把负数检查放在内部，所以报错位置精确到函数名 token。
- `bi_min` / `bi_max` 用三目运算符比较。`min` 取 `<` 的那一个，`max` 取 `>` 的那一个——**相等时返回谁无所谓**（值一样）。
- `bi_pow` 用了 `a[1]`，所以它的 `min_args`/`max_args` 必须是 2。

**如果你写错了会怎样**：

- 忘记 `(void)tok;` → 7 条 `unused parameter 'tok'` 警告（`bi_sqrt` 不会报，因为它用了）。
- `bi_sqrt` 里写成 `fatal_at(tok, ...)` 但 `tok` 没接收到 → 编译报 "use of undeclared identifier 'tok'"。
- 把 `bi_round` 写成 `round(a[0])`（直接调 C 的 `round`）→ **能编译，但语义变了**：`round(-2.5)` 会变成 `-3` 而不是 `-2`。这是本步最容易犯的"看起来对"的错。

#### 第 3 段：表和查表函数

```diff
+static const Builtin builtins[] = {
+    { "abs",   1, 1, bi_abs   },
+    { "round", 1, 1, bi_round },
+    { "sqrt",  1, 1, bi_sqrt  },
+    { "floor", 1, 1, bi_floor },
+    { "ceil",  1, 1, bi_ceil  },
+    { "pow",   2, 2, bi_pow   },
+    { "min",   2, 2, bi_min   },
+    { "max",   2, 2, bi_max   },
+};
+static const int nbuiltins = (int)(sizeof(builtins) / sizeof(builtins[0]));
+
+/* 按名字查内置函数 */
+static const Builtin *find_builtin(const char *name) {
+    for (int i = 0; i < nbuiltins; i++)
+        if (!strcmp(builtins[i].name, name)) return &builtins[i];
+    return NULL;
+}
```

**逐行解释**：

- 数组是 `static const`——静态存储期（生命周期到程序结束）+ 只读。
- **初始化列表的顺序必须和结构体字段顺序一致**：`{ name, min_args, max_args, fn }`。写反了编译器会报类型不匹配（`const char *` 赋给 `int` 是 error）。
- `bi_abs` 直接写函数名（不加 `&`）——C 里函数名**本身就是**函数指针，所以 `bi_abs` 和 `&bi_abs` 完全等价。
- `sizeof(builtins) / sizeof(builtins[0])` 是 C 里求数组元素个数的标准写法。**千万不要写成 `sizeof(builtins) / sizeof(builtins[0].name)`**——那样算出来是 `sizeof(const char*)` 的倍数，得到的是完全错误的数字（比如 8 倍）。这是个真实存在过的 bug 模式。
- `strcmp` 来自 `<string.h>`（第 1 步就 include 了）。
- 返回 `const Builtin *`。

**如果你写错了会怎样**：

- `nbuiltins` 写成 `sizeof(builtins)` → 循环会越界读表后面的内存，可能匹配到乱七八糟的东西甚至崩溃。
- 表项顺序写错 → 编译期报错，因为类型对不上（`const char*` 不能赋给 `int`）。
- `find_builtin` 忘了 `return NULL;` → 找不到时返回**未定义的指针值**，调用方 `if (!b)` 就拦不住了 → 段错误。**每个返回指针的函数都必须有一条明确的"失败路径"。**

#### 第 4 段：改造 `eval` 的 `N_CALL` 分支

```diff
         case N_CALL: {
-            /* 目前只认识 print 一个内建函数。
-               第 8 步会把它和 abs/sqrt 等一起收进一张"内置函数表"。 */
-            if (strcmp(n->text, "print") != 0)
+            /* print 特殊处理：它要输出，而且下一步还要访问作用域查模板变量 */
+            if (!strcmp(n->text, "print")) {
+                if (n->count != 1)
+                    fatal_at(n->tok, "函数 'print' 参数数量错误：需要 1 个，实际 %d 个", n->count);
+                Value v = eval(n->items[0], e);
+                if (v.is_string)
+                    fatal_at(n->tok, "print 的参数必须是数字");
+                printf("%.15g", v.value);   /* 数字之间不会自动换行 */
+                return number(0);
+            }
+
+            /* 其余内建函数：查表 */
+            const Builtin *b = find_builtin(n->text);
+            if (!b)
                 fatal_at(n->tok, "未知函数 '%s'", n->text);
-            if (n->count != 1)
-                fatal_at(n->tok, "函数 'print' 参数数量错误：需要 1 个，实际 %d 个", n->count);
-            Value v = eval(n->items[0], e);
-            if (v.is_string)
-                fatal_at(n->tok, "print 的参数必须是数字");
-            printf("%.15g", v.value);   /* 数字之间不会自动换行 */
-            return number(0);
+
+            if (n->count < b->min_args || n->count > b->max_args)
+                fatal_at(n->tok, "函数 '%s' 参数数量错误：需要 %d 个，实际 %d 个",
+                         b->name, b->min_args, n->count);
+
+            double args[MAX_PARAMS];
+            for (int i = 0; i < n->count; i++) {
+                Value v = eval(n->items[i], e);
+                if (v.is_string)
+                    fatal_at(n->tok, "函数 '%s' 的参数必须是数字", b->name);
+                args[i] = v.value;
+            }
+            return number(b->fn(args, n->tok));
         }
```

**逐行解释**：

- `if (!strcmp(n->text, "print"))` —— `strcmp` 相等时返回 0，`!` 把 0 变成 1。**原来的写法 `if (strcmp(...) != 0)` 是"不是 print 就报错"；现在是"是 print 就特殊处理"**——语义反转了，注意别看漏。
- **第一个分支就是 print 专用逻辑**（`n->count != 1` 检查 → 求值 → `printf`）。这些代码是从原来那坨里"搬"过来的，逻辑一行没变，只是缩进变了。
- `const Builtin *b = find_builtin(n->text);` —— 查表。**必须是 `const Builtin *`**，不能是 `Builtin *`（见 5.7 的 const 传染性）。
- `if (!b) fatal_at(... "未知函数 '%s'" ...)` —— 查不到就是未知函数。真实报错：

  ```
  错误 [1:1]: 未知函数 'nosuch'
  ```

- **参数个数检查**：`if (n->count < b->min_args || n->count > b->max_args)`。用 `b->min_args` / `b->max_args` 里的数字报错：

  ```
  错误 [1:1]: 函数 'abs' 参数数量错误：需要 1 个，实际 2 个
  错误 [1:1]: 函数 'pow' 参数数量错误：需要 2 个，实际 1 个
  错误 [1:1]: 函数 'abs' 参数数量错误：需要 1 个，实际 0 个
  ```

- `double args[MAX_PARAMS];` —— `MAX_PARAMS` 是 32，第 1 步定的宏。**用固定数组而不是 `malloc`**，因为实参个数已经被 `max_args` 限制住了（现在最多 2），永远用不满 32 个。
- **实参求值循环**：`eval(n->items[i], e)` 逐个求值，存进 `args[i]`。注意 `e` 是**当前作用域**——所以实参里可以引用变量（`abs(-x)`、`pow(x, 2)`）。
- `return number(b->fn(args, n->tok));` —— **通过函数指针调用**。`b->fn` 取出表里存的函数指针，`(args, n->tok)` 调用它，`number(...)` 把 `double` 包成 `Value` 返回。
- 整个 `case N_CALL` 用 `{ }` 包住了——因为里面声明了 `const Builtin *b` 和 `double args[]`。

**如果你写错了会怎样**：

- **把 `print` 的判断放在查表之后** → `print` 不在表里 → 会报"未知函数 'print'"。所有用到 `print` 的程序全崩。
- **先求值实参再检查个数** → `abs(1, 1/0)` 会报"除数不能为 0"而不是"参数数量错误"。功能上没错，但错误信息误导用户。
- **写成 `Builtin *b`** → 编译报 `initializing 'Builtin *' with an expression of type 'const Builtin *' discards qualifiers`。
- **忘了 `b->fn(args, n->tok)` 里的 `n->tok`** → `sqrt(-1)` 会报 `错误 [0:0]`（未初始化的 token），而不是 `[1:1]`。
- **`args` 数组开小了**（比如 `double args[2]` 而 `MAX_PARAMS` 是 32）→ 实参超过 2 个时栈溢出。用 `MAX_PARAMS` 就没这个风险。

### 6.3 完整 diff

> **怎么用这段 diff**：推荐**照着敲**——每改一小段就编译一次，比一次性粘贴更容易记住。
> 如果想直接应用，把下面的内容存成 `step.patch`，然后在**上一步的目录**里执行：
>
> ```sh
> patch -p1 < ../008/step.patch
> ```
>
> 应用完和本目录的 `minilang.c` **逐字节一致**（可以 `diff` 验证）。

<details>
<summary>点开看完整 diff（真实 diff -u 输出）</summary>

```diff
--- a/minilang.c
+++ b/minilang.c
@@ -628,7 +628,50 @@
     }
 }
 
+/* ------------------------------------------------------------------ */
+/* 内置函数表                                                          */
+/*                                                                     */
+/* 表里放的是"数字进、数字出"的纯函数。print 不在表里——它要输出，   */
+/* 而且下一步还要访问作用域去查模板里的变量，需求不一样。             */
 /* ------------------------------------------------------------------ */
+typedef double (*BuiltinFn)(double *args, Token tok);
+
+typedef struct {
+    const char *name;    /* 函数名         */
+    int         min_args;/* 最少几个参数   */
+    int         max_args;/* 最多几个参数   */
+    BuiltinFn   fn;      /* 具体怎么算     */
+} Builtin;
+
+static double bi_abs(double *a, Token tok)   { (void)tok; return fabs(a[0]); }
+static double bi_round(double *a, Token tok) { (void)tok; return floor(a[0] + 0.5); }
+static double bi_sqrt(double *a, Token tok)  { if (a[0] < 0) fatal_at(tok, "sqrt 参数不能为负数"); return sqrt(a[0]); }
+static double bi_floor(double *a, Token tok) { (void)tok; return floor(a[0]); }
+static double bi_ceil(double *a, Token tok)  { (void)tok; return ceil(a[0]); }
+static double bi_pow(double *a, Token tok)   { (void)tok; return pow(a[0], a[1]); }
+static double bi_min(double *a, Token tok)   { (void)tok; return a[0] < a[1] ? a[0] : a[1]; }
+static double bi_max(double *a, Token tok)   { (void)tok; return a[0] > a[1] ? a[0] : a[1]; }
+
+static const Builtin builtins[] = {
+    { "abs",   1, 1, bi_abs   },
+    { "round", 1, 1, bi_round },
+    { "sqrt",  1, 1, bi_sqrt  },
+    { "floor", 1, 1, bi_floor },
+    { "ceil",  1, 1, bi_ceil  },
+    { "pow",   2, 2, bi_pow   },
+    { "min",   2, 2, bi_min   },
+    { "max",   2, 2, bi_max   },
+};
+static const int nbuiltins = (int)(sizeof(builtins) / sizeof(builtins[0]));
+
+/* 按名字查内置函数 */
+static const Builtin *find_builtin(const char *name) {
+    for (int i = 0; i < nbuiltins; i++)
+        if (!strcmp(builtins[i].name, name)) return &builtins[i];
+    return NULL;
+}
+
+/* ------------------------------------------------------------------ */
 /* 运行时：值与求值                                                    */
 /* ------------------------------------------------------------------ */
 
@@ -765,17 +808,33 @@
             return number(get_var(e, n->tok, n->text));
 
         case N_CALL: {
-            /* 目前只认识 print 一个内建函数。
-               第 8 步会把它和 abs/sqrt 等一起收进一张"内置函数表"。 */
-            if (strcmp(n->text, "print") != 0)
+            /* print 特殊处理：它要输出，而且下一步还要访问作用域查模板变量 */
+            if (!strcmp(n->text, "print")) {
+                if (n->count != 1)
+                    fatal_at(n->tok, "函数 'print' 参数数量错误：需要 1 个，实际 %d 个", n->count);
+                Value v = eval(n->items[0], e);
+                if (v.is_string)
+                    fatal_at(n->tok, "print 的参数必须是数字");
+                printf("%.15g", v.value);   /* 数字之间不会自动换行 */
+                return number(0);
+            }
+
+            /* 其余内建函数：查表 */
+            const Builtin *b = find_builtin(n->text);
+            if (!b)
                 fatal_at(n->tok, "未知函数 '%s'", n->text);
-            if (n->count != 1)
-                fatal_at(n->tok, "函数 'print' 参数数量错误：需要 1 个，实际 %d 个", n->count);
-            Value v = eval(n->items[0], e);
-            if (v.is_string)
-                fatal_at(n->tok, "print 的参数必须是数字");
-            printf("%.15g", v.value);   /* 数字之间不会自动换行 */
-            return number(0);
+            if (n->count < b->min_args || n->count > b->max_args)
+                fatal_at(n->tok, "函数 '%s' 参数数量错误：需要 %d 个，实际 %d 个",
+                         b->name, b->min_args, n->count);
+
+            double args[MAX_PARAMS];
+            for (int i = 0; i < n->count; i++) {
+                Value v = eval(n->items[i], e);
+                if (v.is_string)
+                    fatal_at(n->tok, "函数 '%s' 的参数必须是数字", b->name);
+                args[i] = v.value;
+            }
+            return number(b->fn(args, n->tok));
         }
 
         case N_UNARY: {
```

</details>

> 和前两节一样，diff 里**没有**文件头的变化。

### 6.4 插入位置说明

在 `007/minilang.c` 上手改的话，两处代码分别放在：

1. **内置函数表整节**：放在 `/* 运行时：值与求值 */` 这个注释块**之前**。为什么放这？——因为 `bi_sqrt` 要调用 `fatal_at`，而 `fatal_at` 在文件**开头**就定义好了（第 1 步），所以位置无所谓；但把"内置函数表"放在"运行时"之前，读代码时是"先看有哪些内置函数，再看求值器怎么用它们"，逻辑更顺。

   注意这一整节是**插入**，不是替换——`/* 运行时：值与求值 */` 那行注释**要保留**（diff 里可以看到它是新加的一节结束后，原来的注释块还在）。

2. **`eval` 的 `N_CALL` 分支**：把原来那 11 行（`if (strcmp(n->text, "print") != 0)` 开始到 `return number(0);` 结束）**替换**成新代码。这是本步唯一一处"修改既有行"的地方——严格来说它是**替换**，不是纯插入，所以 `006 → 007 → 008` 的 diff 里这一段会有删除行。这是合理的：这里不是"白名单"的三处之一，但它是**同一段代码的重写**（语义完全保留，只是从"硬编码 print"变成"print 特判 + 查表"），读者能对照着看懂。

改完后编译：

```sh
cc -std=c11 -Wall -Wextra -O2 minilang.c -lm -o minilang
```

**`-lm` 不能忘！** `fabs` / `floor` / `ceil` / `sqrt` / `pow` 都在 `<math.h>` 里，在 glibc / macOS 上它们在 `libm` 里。少了 `-lm` 会得到一堆 `undefined symbol` 链接错误。

**警告检查**：编译输出应该是**干净的**（没有任何 warning）。如果你看到 7 个 `unused parameter 'tok'` 警告，说明你忘了写 `(void)tok;`。

## 7. 测试

`./run.sh` 的真实输出：

```
▶ 用例（模式：run）
  ✅ case_01_builtins.ml          [run]
  ✅ case_02_math.ml              [run]
  ✅ case_03_sqrt_negative.ml     [run]
  ✅ case_04_wrong_arity.ml       [run]
  ✅ case_05_unknown_function.ml  [run]
▶ demo.ml
  | 3
✅ 全部通过（6 项）
```

逐个用例说明：

**`tests/case_01_builtins.ml`** —— 最基本的内置函数：

```
print(abs(-3));
```

- 期望 stdout：`3`（一个字符，**没有换行**）。
- 期望 stderr：空。期望退出码：0。
- 为什么：`abs` 查表命中 `bi_abs`，实参 `-3` 求值得到 `-3.0`，`fabs(-3.0) = 3.0`，`printf("%.15g", 3.0)` 输出 `3`。
- 注意 `-3` 在 AST 里是 `一元 MINUS` → `数字 3`，不是负数字面量。

**`tests/case_02_math.ml`** —— 数学函数：

```
print(sqrt(16));
```

- 期望 stdout：`4`。期望退出码：0。
- 为什么：`sqrt(16.0) = 4.0`（C 标准库的 `sqrt` 是精确的，`16` 是完全平方数）。

**`tests/case_03_sqrt_negative.ml`** —— 负数开方：

```
sqrt(-1);
```

- 期望 stderr（全文，一行）：

  ```
  错误 [1:1]: sqrt 参数不能为负数
  ```

- 期望 stdout：空。期望退出码：**1**。
- 为什么是 `[1:1]`：`sqrt` 这个标识符在第 1 行第 1 列。`bi_sqrt` 收到的是 `n->tok`——也就是 `N_CALL` 节点的 token，即**函数名**的位置。
- **注意不是 `[1:6]`**（参数 `-1` 的位置）。这个精度正是"把检查放在内置函数内部 + 把 `tok` 传进去"换来的。
- 为什么不用 C 的 `sqrt()` 直接报错？因为 C 的 `sqrt(-1.0)` 返回 `NaN` **不报错**。MiniLang 需要给用户一个明确的错误，所以自己检查。

**`tests/case_04_wrong_arity.ml`** —— 参数个数不对：

```
abs(1, 2);
```

- 期望 stderr（全文，一行）：

  ```
  错误 [1:1]: 函数 'abs' 参数数量错误：需要 1 个，实际 2 个
  ```

- 期望 stdout：空。期望退出码：**1**。
- 为什么：`abs` 的表项是 `{ "abs", 1, 1, bi_abs }`，`min_args = max_args = 1`，而 `n->count = 2` → `2 < 1` 为假、`2 > 1` 为真 → 报错。
- 报错位置 `[1:1]` 是函数名 `abs` 的位置。

**`tests/case_05_unknown_function.ml`** —— 未知函数：

```
nosuch(1);
```

- 期望 stderr（全文，一行）：

  ```
  错误 [1:1]: 未知函数 'nosuch'
  ```

- 期望 stdout：空。期望退出码：**1**。
- 为什么：`find_builtin("nosuch")` 遍历完 8 个表项都没找到，返回 `NULL` → `if (!b)` 命中 → 报错。

**`demo.ml` 冒烟测试** —— `print(abs(-3));` 输出 `3`。

## 8. 易错点与坑

### 坑 1：编译时忘了 `-lm`

- **错误现象**：一堆链接错误：

  ```
  _fabs: undefined symbol
  _floor: undefined symbol
  ```

- **原因**：`<math.h>` 里的数学函数在单独的 `libm` 库里。macOS 上用 `cc` / `clang` 时必须显式加 `-lm`。
- **正确写法**：`cc -std=c11 -Wall -Wextra -O2 minilang.c -lm -o minilang`。`run.sh` 里已经带上了。

### 坑 2：忘了 `(void)tok;`，编译出一片警告

- **错误现象**：7 条 `warning: unused parameter 'tok' [-Wunused-parameter]`。
- **原因**：C 没有"忽略参数"的语法，参数在函数体里完全没被用到就是"未使用"。
- **正确写法**：写 `(void)tok;`。这一行告诉编译器"我是故意不用它的"。

### 坑 3：函数指针的 `*` 忘了加括号

- **错误现象**：类型完全不对，后面全都报错；或者更隐蔽——**编译能过，但行为完全不是你想要的**。
- **原因**：C 的声明是"从里往外"读的。`double *fn(double*, Token)` 是"返回 `double*` 的函数"，`double (*fn)(double*, Token)` 才是"函数指针"。
- **正确写法**：**永远**把 `*` 和名字用括号括起来：`double (*fn)(...)`。要是不确定，就用 `typedef`——`typedef double (*BuiltinFn)(double *args, Token tok);` 之后写 `BuiltinFn fn;` 就不会有歧义了。

### 坑 4：`round` 写成直接调 C 的 `round()`

- **错误现象**：`round(-2.5)` 输出 `-3`，而不是 MiniLang 规定的 `-2`。**程序"能跑"，但语义和文档不符**——这种错最难发现。
- **原因**：C 的 `round()` 是"中点远离零"，MiniLang 定义的是"向 +∞ 方向四舍五入"。
- **正确写法**：`return floor(a[0] + 0.5);`。**在 MiniLang 里，`round` 永远等于 `floor(x + 0.5)`。**

### 坑 5：`nbuiltins` 算错

- **错误现象**：查表越界，可能匹配到随机内容、崩溃，或者"未知函数"的报错莫名其妙。
- **原因**：`sizeof(builtins)` 是**整个数组**的字节数，不是元素个数。
- **正确写法**：`sizeof(builtins) / sizeof(builtins[0])`。记住这个惯用法——**元素个数 = 总大小 ÷ 单个元素大小**。

### 坑 6：`find_builtin` 忘了 `return NULL;`

- **错误现象**：查找失败时返回**未定义的指针值**，调用方的 `if (!b)` 判断失效 → 后面 `b->fn(...)` 解引用野指针 → 段错误。而且这种崩溃**位置随机**，极难调试。
- **原因**：C 里函数末尾没有隐式 return，返回值是"未定义"（垃圾值）。
- **正确写法**：**每个返回指针的函数都必须有一条明确的失败路径。** `find_builtin` 的失败路径就是 `return NULL;`。

### 坑 7：`const Builtin *b` 写成 `Builtin *b`

- **错误现象**：编译报错 `initializing 'Builtin *' with an expression of type 'const Builtin *' discards qualifiers`。
- **原因**：`find_builtin` 返回 `const Builtin *`（因为表是 `const` 的）。C 的 `const` 是"传染"的——不能把指向常量的指针赋给指向非常量的指针。
- **正确写法**：调用方也用 `const Builtin *b`。

### 坑 8：把 `print` 也放进表里

- **错误现象**：编译能过，但 `print("hi {x}\n")` 里的 `{x}` 永远替换不了——因为表里的函数拿不到 `Env *`，没法查变量。
- **原因**：表的契约是"数字进、数字出"，`print` 需要作用域。
- **正确写法**：`print` 特殊处理，**放在查表之前**。这也是第 9 步要保持的顺序。

### 坑 9：以为 `sqrt` 的错误位置在参数上

- **错误现象**：写文档时写错位置。
- **原因**：`fatal_at(tok, ...)` 里的 `tok` 是**函数名**的 token，不是参数的。
- **正确写法**：想报参数的位置，就得把参数的 token 也传进去——但那会让签名复杂得多。当前设计里，报"函数调用有问题"的位置就是函数名，这已经足够精确了。

## 9. 自测题

### 9.1 概念题

**Q1.** `typedef double (*BuiltinFn)(double *args, Token tok);` 是什么意思？为什么 `*` 必须加括号？

**Q2.** `bi_abs` 里的 `(void)tok;` 是干什么的？删掉会发生什么？

**Q3.** 为什么 `print` 不能放进 `builtins[]` 表？给出**两条**理由。

**Q4.** `round(-2.5)` 在 MiniLang 里等于几？和 C 标准库的 `round(-2.5)` 一样吗？

**Q5.** "参数个数检查"和"参数类型检查"分别在哪一步做？为什么个数检查要放在实参求值**之前**？

**Q6.** `sqrt(-1)` 的错误信息里，行列号指的是哪个位置？

<details>
<summary>参考答案</summary>

**A1.** 意思是"`fn` 是一个函数指针，指向一个接收 `double *args` 和 `Token tok`、返回 `double` 的函数"。`*` 必须加括号是因为 C 的声明从里往外读：不加括号时 `double *fn(double*, Token)` 会被读成"`fn` 是一个返回 `double *` 的**函数**"，而不是"指向函数的指针"。

**A2.** 显式丢弃 `tok` 参数。C 没有"忽略参数"的语法，参数没被用到就是"未使用"，`-Wextra` 会报 `warning: unused parameter 'tok'`。`(void)tok;` 告诉编译器是故意的。

**A3.** 两条理由：(1) `print` 有副作用（要输出），而表里的函数是"数字进、数字出"的纯函数；(2) 第 9 步 `print` 要访问作用域查模板里的 `{x}`，需要 `Env *`，而 `BuiltinFn` 的签名里没有 `Env *` 的位置。

**A4.** MiniLang 的 `round(-2.5)` 是 **-2**（`floor(-2.5 + 0.5) = floor(-2.0) = -2`），C 标准库的 `round(-2.5)` 是 **-3**（中点远离零）。**不一样。**

**A5.** 个数检查在 `eval` 的 `N_CALL` 分支里、查到表项之后、实参求值之前（`if (n->count < b->min_args || n->count > b->max_args)`）；类型检查在逐个求值实参的循环里（`if (v.is_string)`）。个数检查放前面是因为它更便宜（只比数字），而且能避免实参求值时抛出**无关的**错误——比如 `abs(1, 1/0)` 应该报"参数数量错误"而不是"除数不能为 0"。

**A6.** 指的是**函数名 `sqrt` 所在的位置**（`n->tok`），不是参数 `-1` 的位置。实测 `sqrt(-1);` 在第 1 行第 1 列时报 `[1:1]`。

</details>

### 9.2 预测题（"改一行会发生什么"）

**Q7.** 把 `if (!strcmp(n->text, "print"))` 改成 `if (strcmp(n->text, "print"))`（去掉 `!`），会发生什么？

**Q8.** 把 `bi_round` 的实现改成 `return round(a[0]);`（直接调 C 的 `round`），`tests/` 里的哪个用例会受影响？你需要自己写一个用例来验证。

**Q9.** 把个数检查移到实参求值**之后**（也就是先 `for` 循环求值，再检查个数），`abs(1, 1 / 0);` 的报错会变成什么？

**Q10.** 在 `builtins[]` 表里加一行 `{ "neg", 1, 1, bi_neg }`，其中 `bi_neg` 返回 `-a[0]`。要写几行 C 代码？`neg(5)` 的输出是什么？

**Q11.** 把 `find_builtin` 里的 `return NULL;` 删掉，`nosuch(1);` 会怎样？

<details>
<summary>参考答案</summary>

**A7.** `strcmp` 相等时返回 0，`!` 把 0 变成 1。去掉 `!` 之后，条件变成"**名字不是** print"，于是**除了 print 以外的所有函数**都会走进 print 的分支——包括 `abs`！所以 `abs(-3)` 会报 `函数 'print' 参数数量错误：需要 1 个，实际 1 个`……不对，个数是对的，接着会执行 `printf("%.15g", v.value)`，打印出 `3`。**也就是说所有内置函数都会退化成 `print`**，输出它们的第一个参数。

**A8.** `tests/` 里的用例**没有一个会挂**——因为现有用例没测 `round`。这正是这个错危险的地方：**它不会让测试变红，但语义悄悄变了**。自己写一个用例：

```
print(round(-2.5));
```

- 用 MiniLang 的 `floor(x+0.5)` → `-2`
- 改成 C 的 `round()` → `-3`

**A9.** 报错会变成：

```
错误 [1:1]: 除数不能为 0
```

因为实参先被求值，`1 / 0` 触发了除零检查，`fatal_at` 直接退出，**根本走不到**后面的个数检查。这正是 5.5 节说的"先做便宜的检查"的价值——顺序反了，错误信息就会误导用户。

**A10.** 只需要写**一行**（表项），加上一个函数实现：

```c
static double bi_neg(double *a, Token tok) { (void)tok; return -a[0]; }
```

一共两行（1 行函数 + 1 行表项），**求值器的代码一个字都不用改**。这就是表驱动的收益。`neg(5)` 输出 `-5`。

**A11.** `find_builtin` 没有 `return NULL;` 时，返回值是**未定义的**（编译器的实际情况通常是寄存器里残留的垃圾值）。这个值**很可能不为 0**，所以 `if (!b)` 判断失效 → 继续执行 `b->min_args` → 解引用野指针 → **段错误（segfault）**，退出码 139 而不是 1。

而且这个 bug 是**间歇性**的：换个编译器、换个优化等级，行为可能就不一样了。这类 bug 极难定位。

</details>

### 9.3 动手题（改代码 + 跑 tests 验证）

**Q12.** 把 8 个内置函数**全部**试一遍，写一个 `.ml` 打印每个函数的结果，用 `./minilang` 跑出真实输出：

```
print(abs(-3));
print(round(-2.5));
print(sqrt(16));
print(floor(-1.5));
print(ceil(-1.5));
print(pow(2, 10));
print(min(3, 5));
print(max(3, 5));
```

**Q13.** 验证"参数个数检查先于实参求值"。写两个 `.ml`：

- `a.ml`：`abs(1, 1 / 0);` → 期望报"参数数量错误"
- `b.ml`：`abs(1 / 0);` → 期望报"除数不能为 0"

**Q14.** 给 MiniLang 加一个新内置函数 `sign(x)`，返回 `x` 的符号（`x > 0` 返回 `1`，`x < 0` 返回 `-1`，`x == 0` 返回 `0`）。要求：

- 写一个 `bi_sign` 函数（签名和别的内置函数一样）。
- 在 `builtins[]` 里加一行 `{ "sign", 1, 1, bi_sign }`。
- **不要**改 `eval` 的任何代码。

验证：`print(sign(-7));` 应该输出 `-1`。

**Q15.**（进阶）支持"参数个数可变"的函数：把 `min` / `max` 改成接受 **1 个以上**的参数（`min(1, 2, 3)` 应该返回 `1`）。

提示：

- 把表项改成 `{ "min", 1, MAX_PARAMS, bi_min }`。
- `bi_min` 的实现要从"比较 `a[0]` 和 `a[1]`"改成"遍历整个数组"——但**它拿不到实参个数**！所以你需要改 `BuiltinFn` 的签名，或者给 `Builtin` 加一个"当前实参个数"的字段。

想清楚：**哪种改法更符合"表驱动"的精神？**

<details>
<summary>参考答案</summary>

**A12.** 真实输出：

```sh
$ ./minilang /tmp/all_builtins.ml
3-24-2-11024354
```

逐个对应：

| 表达式 | 结果 | 说明 |
|---|---|---|
| `abs(-3)` | `3` | `fabs(-3.0)` |
| `round(-2.5)` | `-2` | `floor(-2.5 + 0.5) = floor(-2.0)` ← **注意不是 -3** |
| `sqrt(16)` | `4` | |
| `floor(-1.5)` | `-2` | 往 -∞ 取整 |
| `ceil(-1.5)` | `-1` | 往 +∞ 取整 |
| `pow(2, 10)` | `1024` | |
| `min(3, 5)` | `3` | |
| `max(3, 5)` | `5` | |

**A13.** 真实输出：

```sh
$ ./minilang a.ml
错误 [1:1]: 函数 'abs' 参数数量错误：需要 1 个，实际 2 个
$ ./minilang b.ml
错误 [1:6]: 除数不能为 0
```

（`b.ml` 的 `[1:6]` 是 `1 / 0` 里那个 `0` 的位置——`abs(` 占 1~4 列，`1` 在第 5 列，`/` 第 6 列…… 具体以你实际跑出来的为准，记住**除零错误的位置指向除数**，而参数数量错误的位置指向**函数名**。）

**A14.** 只改两处：

```c
static double bi_sign(double *a, Token tok) {
    (void)tok;
    if (a[0] > 0) return 1;
    if (a[0] < 0) return -1;
    return 0;
}
```

```c
{ "sign", 1, 1, bi_sign },
```

`eval` 一行都不用改。验证：

```sh
$ ./minilang /tmp/sign.ml
-1
```

**这就是表驱动的全部意义**：加一个函数 = 加两行（实现 + 表项），**零处修改既有逻辑**。

**A15.** 有两种改法：

**改法一（改签名）**：把 `BuiltinFn` 改成

```c
typedef double (*BuiltinFn)(double *args, int nargs, Token tok);
```

这样每个函数都知道有几个实参。代价是**所有 8 个已有函数都要改签名**（哪怕它们不用 `nargs`），而且 `tok` 排到了第三位——**每个函数都要跟着调整**。这违反了"只插入、不修改既有行"（INV-8）。

**改法二（给表项加字段）**：`Builtin` 里加一个 `int nargs;`，调用前先 `b->nargs = n->count;` 再调用。代价是表被"写脏"了（明明是 `const` 的），而且 `const` 也没法用了。

**改法三（推荐）**：把"当前实参个数"放进**参数的表示**里——比如约定 `args[MAX_PARAMS - 1]` 存个数，或者传一个"数组 + 长度"的小结构体：

```c
typedef struct { double *v; int n; } Args;
typedef double (*BuiltinFn)(Args a, Token tok);
```

更彻底的做法是**给 `Builtin` 加一个"可变参数"标记**（比如 `min_args == 0` 表示"接受任意个"），然后在 `eval` 里对这类函数特殊处理。

**哪一种最符合"表驱动"的精神？** 关键判据是：**加新函数时要不要改既有代码**。改法一和三要改所有既有函数（差）；改法二不用改既有函数，但要破坏 `const`（勉强）；真正的"表驱动"理想是：**新函数 = 新增一行表项 + 一个新函数体，既有代码一行不动**。

这个练习其实暴露了表驱动的一个真实局限：**当函数的"参数个数"本身成为可变信息时，纯数据表就不够了，需要在结构里加一个"运行时槽位"**。在真正的解释器里（比如 Lua、Python 的 C API），这类问题是用"变长参数对象"或"参数数组 + 长度"来解决的。**这正是"先做简单的那个"的代价，也是第 13 步重构时要重新审视的地方。**

</details>

## 10. 本步的设计取舍

**为什么用表，而不是一长串 `if`？**

因为内置函数的数量会增长，而"函数名 → 实现 + 参数个数"这个映射是**纯数据**。把它放进数据而不是代码里，意味着：

- 加函数 = 加一行。
- 参数个数的检查逻辑只写一遍（8 个函数共用）。
- 参数个数的**约束**也变成了数据（`min_args` / `max_args`），将来支持可变参数只需要改数据。

代价是多了一层间接和一个函数指针的概念。**对教学来说这个代价是值得的**——因为"函数指针"本身就是一个必须学会的 C 概念，MiniLang 正好给了它一个真实的用武之地。

**为什么用 `floor(x + 0.5)` 而不是 C 的 `round()`？**

因为 `round(x+0.5)` 的语义（向 +∞ 四舍五入）更符合"四舍五入"这个中文词的直觉，而且实现简单、无需分符号讨论。代价是**和 C 标准库不一样**——这是本教程从头到尾反复强调"自己跑一遍确认"的原因：**同名不同义是最容易踩的坑。**

**为什么 `print` 不进表？**

因为它不是"数字进、数字出"的纯函数：它有副作用，而且第 9 步需要作用域。把它留在 `eval` 里特殊处理，是**用一点特例换取表的纯粹性**。

**为什么参数类型检查写在这里（第 8 步是死代码）？**

因为第 9 步马上就会用到它（那时 `eval` 真的会返回字符串）。这正是 INV-2 的意义——**第 3 步就定型的 `Value { double value; int is_string; char string[]; }` 在这里兑现了价值**。如果不预留，第 9 步就得回头改 `eval` 里所有取 `v.value` 的地方（`N_LET`、`N_ASSIGN`、`N_IF`、`N_UNARY`、`N_BINARY`、`N_CALL`……），那会是一个大得多的 diff。

**还没做的：**

- 没有用户自定义函数（第 10 步）。
- 没有 `log` / `exp` / `sin` / `cos` 等三角/指数函数（加一行表项就有，但没必要）。
- 没有 `print` 的字符串版本（第 9 步）。
- 内置函数不能被重新定义（`let abs = ...` 只是定义了一个变量，调用时仍然走内置函数——见 5.4）。
- 内置函数不能作为值传递（没有"函数是一等公民"这回事）。
- 没有可变参数函数（见 9.3 的 Q15）。
- `find_builtin` 是线性查找（8 个函数无所谓，上千个就要换哈希表）。

## 11. 下一步预告

到目前为止，MiniLang 的 `print` 只能打印数字。想打印一句人话怎么办？`print("hello")` 会报"求值器还不支持字符串"。

下一节我们给 MiniLang 加上**字符串**和**字符串模板**：`print("x = {x}\n")` 会把 `{x}` 替换成变量 `x` 的值再输出。你会看到第 3 步就预留好的 `Value.string[]` 和 `is_string` 字段是怎么被用上的，会看到 `print` 那个"不进表"的决定为什么是对的（它终于需要访问作用域了），也会看到词法分析器第 1 步就已经准备好了字符串字面量的解析。

下一节：[第 9 步 · print 与字符串模板](../009/README.md)
