# 第 5 步 · 语句 / let / 作用域

> 进度 ████░░░░░░ 5/13 ｜ 上一节：[第 4 步 · 比较与逻辑运算](../004/README.md) ｜ 下一节：[第 6 步 · if / else](../006/README.md)

## 1. 本步目标

读完你将能够：

- 让程序真正**跑起来**：`run` 模式从"尚未实现"变成能执行一整段程序。
- 区分**语句（statement）**和**表达式（expression）**：语句做一件事，表达式算出一个值。
- 用 `let` 定义变量、用 `x = ...` 赋值，并且知道这两者的区别。
- 理解**块级作用域**：`{ }` 里的变量出了块就没了，同名时内层遮蔽外层。
- 解释"为什么变量表要做成动态数组"——这一条在**第 11 步**会闭环（递归到第 100 层不崩，
  而固定数组版本第 8 层就段错误）。
- 理解 `Flow` 结构是**提前定型**的：第 11 步加 `return` 时，`exec()` 的签名一个字都不用改。

## 2. 运行方式

```sh
cd 005
./run.sh                 # 编译 + 跑全部用例 + 跑 demo
./minilang demo.ml       # 默认就是 run 模式，本步它终于能用了
./minilang demo.ml ast   # 打印语法树
./minilang demo.ml tokens
```

本步 `run.sh` 的真实输出见 [第 7 节](#7-测试)。

## 3. 现在的程序能做什么

| 模式 | 输入 | 输出 | 退出码 |
|---|---|---|---|
| `tokens` | 任意 `.ml` | token 列表 | 0 / 1 |
| `ast` | 任意 `.ml` | **整个程序**的语法树 | 0 / 1 |
| `run`（默认） | 任意 `.ml` | **执行程序**的输出 | 0 / 1 |

前四步 `ast` 模式处理的是"一个表达式"，`run` 模式是空的。
从这一步起，两者的输入都变成了"**一整个程序**"（一串语句）。

`005/demo.ml` 的内容：

```
// 第 5 步：变量、赋值和作用域
let x = 10;
x = x + 5;
{
    let x = 100;   // 块里的 x 遮蔽了外面的 x
    print(x);
}
print(x);           // 这里还是 15
```

真实运行：

```sh
$ ./minilang demo.ml
10015
```

注意输出是 `10015` 而不是分三行的 `100` / `15`——**`print` 不会自动换行**，
数字之间是紧挨着的。第 9 步有了字符串模板，就可以用 `\n` 换行了。

## 4. 前置知识回顾

上一步我们把"一个表达式"算成了值。现在要把粒度从**表达式**扩大到**语句**。

```text
表达式（expression）  →  算出一个值    1 + 2 * 3   →  7
语句（statement）     →  做一件事      x = x + 1;
```

C 语言里这个区分是硬性的：`int y = 1 + 2;` 左边是声明，右边是表达式；
`1 + 2;` 单独一条是合法的语句（把结果丢掉），但 `x + 1` 不是合法的语句（缺分号）。

MiniLang 简化成三条规则：

| 写法 | 类别 | 说明 |
|---|---|---|
| `let x = 表达式;` | 语句 | 定义变量 |
| `x = 表达式;` | 语句 | 赋值 |
| `表达式;` | 语句 | 表达式语句（结果丢掉） |
| `{ 语句 ... }` | 语句 | 代码块 |

**所有语句都以 `;` 结尾**（代码块本身不用）。

## 5. 本步引入的核心概念

### 5.1 概念一：语句的语法分析

前四步的语法分析器只会处理表达式。第 5 步要加一个 `statement()`。

先看整体形状：

```
statement()
├── '{'          → block()      代码块
├── 'let'        → N_LET        定义变量
├── 标识符 '='   → N_ASSIGN     赋值
└── 其它         → N_EXPR       表达式语句
```

每一条语句都对应一个 AST 节点类型：

| 节点 | 存什么 |
|---|---|
| `N_LET` | `text` = 变量名，`a` = 初始值的表达式 |
| `N_ASSIGN` | `text` = 变量名，`a` = 新值的表达式 |
| `N_EXPR` | `a` = 表达式 |
| `N_BLOCK` | `items[]` = 里面的语句列表 |

```c
if (match(T_LET)) {
    Node *n = node(N_LET, t);
    Token id = expect(T_ID, "let 后需要变量名");
    snprintf(n->text, sizeof(n->text), "%s", id.text);
    expect(T_ASSIGN, "变量定义需要 '='");
    n->a = expression();
    expect(T_SEMI, "语句末尾需要 ';'");
    return n;
}
```

**代码要点**：`node()` 里的 `memset(n, 0, sizeof(*n))` 保证 `n->a`、`n->text`
这些没被赋值的字段是干净的零值。这在第 6 步加 `if` 时会变得非常重要
（`n->c` 没有 `else` 时就是靠这个 `memset` 变成 `NULL` 的）。

**边界情况**：`let` 后面如果跟的不是标识符（比如 `let 1 = 2;`），
`expect(T_ID, ...)` 会报错"语法错误：let 后需要变量名，当前是 '1'"。

### 5.2 概念二：赋值语句怎么和表达式语句区分

这是本步最容易写错的地方。

赋值语句长得像表达式语句：`x = 1;` 和 `f();` 在语法上都是"某个东西加分号"。
怎么区分？

```c
/* 赋值语句：标识符 = 表达式; */
if (cur()->type == T_ID && toks[pos + 1].type == T_ASSIGN) {
    Token id = *cur();
    pos += 2;                       /* 跳过标识符和 '=' */
    ...
}
```

判断条件是**两个**：

1. 当前 token 是标识符（`T_ID`）；
2. **下一个** token 是单个等号 `T_ASSIGN`。

第二个条件不能写成"下一个 token 是等号"——因为 `==`（相等比较）在词法层面
是**另一个** token 类型 `T_EQ`，所以这里天然不会把 `x == 1;` 误判成赋值。

```text
x = 1;      →  T_ID, T_ASSIGN, ...   → 赋值语句
x == 1;     →  T_ID, T_EQ, ...       → 表达式语句（比较）
x;          →  T_ID, T_SEMI          → 表达式语句（引用变量）
```

**如果你写错了会怎样**：如果只判断"当前是标识符"，那 `x == 1;` 会被当成
`x = (= 1)`，然后 `expect(T_ASSIGN)` 会看到 `T_EQ` 于是报
"语法错误：赋值需要 '='，当前是 '=='"——错误信息很难懂，因为真正的问题是判断条件写松了。

**边界情况**：`toks[pos + 1]` 会不会越界？
不会——词法分析器总是会在最后补一个 `T_EOF` token，而 `T_EOF` 不是 `T_ID`，
所以 `pos + 1` 最多指到那个 `EOF`，永远在数组范围内。

### 5.3 概念三：作用域链（Env 链）

变量存在哪儿？答案是**一串环境（Environment）串起来的链**：

```text
        ┌───────────────┐
        │ global        │  parent = NULL
        │  x = 10       │
        └───────┬───────┘
                │ parent
        ┌───────▼───────┐
        │ block          │  ← 每进入一个 { } 就 new 一个
        │  x = 100      │
        └───────────────┘
```

`Env` 结构体：

```c
struct Env {
    Var   *vars;        /* 变量表，按需增长（不是固定长度数组！） */
    int    count;       /* 已经定义了多少个变量 */
    int    cap;         /* 变量表当前能放多少个 */
    struct Env *parent; /* 外层作用域，NULL 表示最外层 */
};
```

查找变量时**从内往外找，找到第一个就用**：

```c
static double get_var(Env *e, Token t, const char *name) {
    for (Env *p = e; p; p = p->parent) {
        int i = find_var(p, name);
        if (i >= 0) return p->vars[i].value;
    }
    fatal_at(t, "未定义变量 '%s'", name);
    return 0;
}
```

查找过程（以 `x` 为例）：

```text
get_var(block, "x")
   │
   ├─ find_var(block,  "x")  → 找到了！  返回 100
   │
   └─（如果 block 里没有）继续往外：
      find_var(global, "x")  → 找到了   返回 10
```

**这就叫"遮蔽"（shadowing）**：内层定义的 `x` 把外层的 `x` 挡住了。

### 5.4 概念四：`find_var` 为什么要倒着找

```c
static int find_var(Env *e, const char *name) {
    for (int i = e->count - 1; i >= 0; i--)
        if (!strcmp(e->vars[i].name, name)) return i;
    return -1;
}
```

变量表是一个**数组**（顺序存放），同一个作用域里如果有两个同名变量，
后写的应该"赢"——所以要**从后往前找**。

不过本步的 `define_var` 会阻止同作用域重复定义：

```c
static void define_var(Env *e, Token t, const char *name, double v) {
    if (find_var(e, name) >= 0)
        fatal_at(t, "变量 '%s' 在当前作用域中已定义", name);
    env_push(e, name, v);
}
```

所以"倒着找"在本步其实用不上，但它是**正确的写法**——第 10 步的函数参数、
第 11 步的递归都会依赖"后定义者优先"这条规则。

**注意**：`define_var` 只检查**当前作用域**（`find_var(e, ...)` 只看 `e` 自己），
所以内层 `let x` 可以遮蔽外层的 `x`，这是**允许**的；
但同一个作用域里 `let x` 两次会报错。

### 5.5 概念五：赋值和定义走的是两条路

| 操作 | 函数 | 作用域 | 找不到会怎样 |
|---|---|---|---|
| 定义 `let x = 1` | `define_var` | **只在当前作用域** | 重复定义 → 报错 |
| 赋值 `x = 1` | `set_var` | **沿链往外找，找到就改** | 找不到 → 报错 |

```c
static void set_var(Env *e, Token t, const char *name, double v) {
    for (Env *p = e; p; p = p->parent) {
        int i = find_var(p, name);
        if (i >= 0) { p->vars[i].value = v; return; }   /* 改外层的也行 */
    }
    fatal_at(t, "不能给未定义变量 '%s' 赋值", name);
}
```

所以这两段代码的行为**完全不同**：

```
let x = 1;      →  在当前作用域新建 x
x = 2;          →  找到最近的那个 x，改它的值
```

**边界情况**：给一个从来没定义过的变量赋值，报的是
"不能给未定义变量 'y' 赋值"，而不是"未定义变量 'y'"——
因为"赋值"这个动作本身隐含了"它应该已经存在"的预期。

### 5.6 概念六：为什么变量表用动态数组，而不是 `Var vars[2048]`

这是本步**最值得记住**的一个决定。当时看起来只是"顺手这么写了"，
但它决定了后面能不能写递归。

先算账。固定数组版本：

```c
typedef struct { char name[MAX_NAME]; double value; } Var;   /* 128 + 8 = 136 字节 */
typedef struct { Var vars[2048]; int count; struct Env *parent; } Env;
```

`sizeof(Env)` = `2048 × 136 + 16` ≈ **272 KB**。

而 `Env` 是**C 栈上的局部变量**：

```c
case N_CALL: {
    Env local;                     /* ← 272 KB，压在栈上 */
    ...
    Flow flow = exec(d->a, &local);   /* ← 函数体又是一个代码块，再压一个 */
```

**每递归一层要压两个 `Env`**（调用处一个 + 函数体的代码块一个），
约 545 KB/层。macOS 的默认栈是 8 MB，所以：

```text
8 MB ÷ 545 KB ≈ 15 层
```

也就是说，如果第 5 步用了固定数组，**递归到第 8~15 层就会段错误**，
`MAX_CALL_DEPTH = 128` 这个保护根本用不上。

动态数组版本：`sizeof(Env)` 只有 **24 字节**（一个指针 + 两个 int + 一个指针），
栈上放多少层都没关系，变量本身在堆上。

```c
static void env_push(Env *e, const char *name, double v) {
    if (e->count >= MAX_VARS)
        fatal_at((Token){.line = 0, .col = 0}, "变量数量超出上限");
    if (e->count == e->cap) {
        int ncap = e->cap ? e->cap * 2 : 8;      /* 容量翻倍 */
        Var *nv = (Var *)realloc(e->vars, (size_t)ncap * sizeof(Var));
        if (!nv) { perror("realloc"); exit(1); }
        e->vars = nv;
        e->cap  = ncap;
    }
    snprintf(e->vars[e->count].name, sizeof(e->vars[e->count].name), "%s", name);
    e->vars[e->count].value = v;
    e->count++;
}
```

**代码要点**（都是 C 里的经典坑）：

1. `realloc` **可能失败**（返回 `NULL`），失败时原来的指针还在，
   但内容没复制过去——所以必须检查，并且**用返回值**而不是原来的指针：

   ```c
   Var *nv = realloc(e->vars, ncap * sizeof(Var));   /* ✅ */
   e->vars = realloc(e->vars, ncap * sizeof(Var));   /* ❌ 失败时旧指针也丢了，内存泄漏 */
   ```

2. 容量**翻倍**（`cap * 2`）而不是"每次 +1"，这样插入 n 个变量的总拷贝次数是
   O(log n) 次摊销，不是 O(n) 次。

3. `MAX_VARS` 的检查放在扩容**之前**，否则会先分配一块巨大的内存再报错。

**本步只是打个伏笔**——第 11 步讲递归的时候，我们会**真的跑一遍**这个对照实验，
亲眼看到固定数组版本在第 8 层段错误、动态数组版本跑到第 100 层都没事。

### 5.7 概念七：`Flow` 为什么要提前定型

第 11 步要加 `return`。`return` 会让一条语句**提前结束**并**带回一个值**，
这个信息必须能沿着调用链往上传。

如果第 5 步的 `exec()` 返回 `void`，第 11 步就得改成：

```c
static Flow exec(Node *n, Env *e);   /* 签名变了！ */
```

而**所有调用点都要跟着改**——`exec` 在代码里有 7 处调用，一处漏掉就编译不过。
更糟的是，这种"改签名"会污染 diff，让读者以为第 11 步动了很多东西。

所以第 5 步就把它定型：

```c
/* 控制流信号：一条语句执行完可能什么都不返回，也可能提前"return"。
   returned 字段从现在起就定型了，后面加 return 语句时不用改 exec 的签名。 */
typedef struct {
    int   returned;  /* 1 表示这条语句链里有 return  */
    Value value;     /* return 带回来的值          */
} Flow;

static Flow noflow(void) {
    Flow f;
    f.returned = 0;
    f.value = number(0);
    return f;
}
```

现在 `exec()` 永远返回 `Flow`，只是 `returned` 永远是 0。
第 11 步只需要在 `case N_RETURN:` 里把 `returned` 置 1，**签名一个字都不用改**。

### 5.8 概念八：`eval` 和 `exec` 为什么要分成两个函数

前四步只有一个 `eval`。现在有了语句，必须分开：

| 函数 | 输入 | 输出 | 干什么 |
|---|---|---|---|
| `eval(Node *n, Env *e)` | 表达式节点 | `Value` | 算出一个值 |
| `exec(Node *n, Env *e)` | 语句节点 | `Flow` | 做一件事 |

```text
表达式树                        语句树
   │                              │
   ▼                              ▼
eval() ──► Value              exec() ──► Flow
   │                              │
   └── 里面可能调用 eval() ────────┘
```

注意 `exec` 处理 `N_LET` 时会调用 `eval` 算右边那个表达式：

```c
case N_LET: {
    Value v = eval(n->a, e);        /* 先算值 */
    if (v.is_string) fatal_at(n->tok, "变量目前只支持数字类型");
    define_var(e, n->tok, n->text, v.value);   /* 再存起来 */
    return f;
}
```

**为什么要分开而不合并成一个函数**？因为两者的语义完全不同：

- 表达式**有值**，`eval` 必须返回一个 `Value`；
- 语句**没值**（`x = 1;` 执行完你不会得到什么），
  它只有"要不要把结果往上传"这一个信息。

如果硬塞进一个函数，就只能返回一个"既是值又是流"的万能类型，
或者到处判断"这个节点到底是表达式还是语句"——两种都很糟。

**这也解释了 INV-4**：从这一步起 `eval` / `exec` 物理分离，后面再没合并过。

## 6. 动手：把代码加进来

### 6.1 改动清单

| 位置 | 改了什么 | 行数 | 为什么 |
|---|---|---|---|
| 宏定义区 | 加 `MAX_VARS`、`MAX_NAME` | +2 | 变量表上限、变量名长度上限 |
| `NodeType` 枚举 | 加 `N_BLOCK`/`N_LET`/`N_ASSIGN`/`N_EXPR` | +4 | 四种新语句 |
| 语法分析器 | 加 `statement()` / `block()` / `parse_program()` | +65 | 语句的解析 |
| `dump_ast()` | 加四个 `case` | +16 | 打印新语句 |
| 运行时 | 加 `Var`/`Env` 结构、`env_init`/`env_free`/`env_push`/`find_var`/`get_var`/`define_var`/`set_var` | +90 | 作用域与变量 |
| 运行时 | 加 `Flow` 结构、`noflow()` | +14 | 提前定型的控制流信号（INV-3） |
| `eval()` | `N_VAR` 改成查作用域；`N_CALL` 支持 `print` | +14 | 变量引用 + 第一个内建函数 |
| `eval()` | 加 `default:` 分支 | +4 | 语句节点不能出现在表达式里 |
| 运行时 | 加 `exec()` | +46 | 执行语句 |
| `mode_ast` / `mode_run` | 改成解析并执行整个程序 | +20 | `run` 模式终于能用 |

合计约 **+270 行**。

### 6.2 分段讲解

#### 6.2.1 新的 AST 节点类型

```diff
     N_VAR,      /* 变量引用          */
     N_UNARY,    /* 一元运算：-x !x   */
     N_BINARY,   /* 二元运算：a + b   */
-    N_CALL      /* 函数调用：f(a,b)  */
+    N_CALL,     /* 函数调用：f(a,b)  */
+    N_BLOCK,    /* 代码块 { ... }    */
+    N_LET,      /* let x = 1;        */
+    N_ASSIGN,   /* x = 1;            */
+    N_EXPR      /* 表达式语句        */
 } NodeType;
```

**逐行解释**：注意 `N_CALL` 后面补了一个逗号——枚举的最后一个元素后面**不能**有逗号
（C99 之后允许，但这里保持 C89 风格更保险）。

**如果你写错了会怎样**：漏掉逗号 → 编译报"expected '}'"。多写一个逗号 → 某些老编译器报错。

**结构体一个字没改**——`a`/`b`/`c` 三个通用子指针和 `items[]` 数组足够表达这四种语句，
这正是 INV-5 "表结构一步定型"的意思。

#### 6.2.2 语句解析器

```c
/* 代码块：{ 语句 语句 ... } */
static Node *block(void) {
    Token t = expect(T_LB, "代码块需要 '{'");
    Node *n = node(N_BLOCK, t);
    while (cur()->type != T_RB && cur()->type != T_EOF) {
        if (n->count >= 128) fatal_at(*cur(), "单个代码块语句过多（上限 128）");
        n->items[n->count++] = statement();
    }
    expect(T_RB, "代码块缺少 '}'");
    return n;
}
```

**逐行解释**：

- `expect(T_LB, ...)` 消费掉 `{`，并在不是 `{` 的时候报错。
- 循环条件里同时检查 `T_RB` 和 `T_EOF`：如果源码写到一半就没了（比如文件末尾），
  循环必须停下来，否则会一直读越界。停下来之后 `expect(T_RB, "代码块缺少 '}'")`
  会给出准确的报错。
- `n->items[n->count++] = statement();` —— 语句列表存在 `items[]` 里，
  和函数调用的实参列表共用同一个数组。

**如果你写错了会怎样**：只判断 `T_RB` 不判断 `T_EOF` 的话，
遇到 `{` 但没有 `}` 的文件会**死循环**（读到数组外面去），而不是报错。

```c
/* 赋值语句：标识符 = 表达式;
   这里要特别小心不要把 == 当成 =：
   靠"当前是标识符，且下一个 token 是 T_ASSIGN"来判断。 */
if (cur()->type == T_ID && toks[pos + 1].type == T_ASSIGN) {
    Token id = *cur();
    pos += 2;
    Node *n = node(N_ASSIGN, id);
    snprintf(n->text, sizeof(n->text), "%s", id.text);
    n->a = expression();
    expect(T_SEMI, "赋值语句末尾需要 ';'");
    return n;
}
```

**逐行解释**：`pos += 2` 一次跳过两个 token（标识符和 `=`），
比写两次 `pos++` 更不容易漏。

**如果你写错了会怎样**：判断条件写成 `cur()->type == T_ID`（漏掉第二个条件），
那么 `x == 1;` 会被当成赋值，然后报一个很难懂的"语法错误：赋值需要 '='，当前是 '=='"。

#### 6.2.3 整个程序

```c
/* 整个程序：一串语句 */
static Node *parse_program(void) {
    Node *n = node(N_BLOCK, toks[0]);
    while (cur()->type != T_EOF) {
        if (n->count >= 128) fatal_at(*cur(), "顶层语句过多（上限 128）");
        n->items[n->count++] = statement();
    }
    return n;
}
```

**思路**：整个程序本质上就是一个"装着所有顶层语句的代码块"。
复用 `N_BLOCK` 节点，`mode_run` 里遍历 `prog->items` 逐条执行就行。

**如果你写错了会怎样**：`node(N_BLOCK, toks[0])` 的 token 用的是 `toks[0]`
（文件第一个 token），这样出错的行列号是"文件开头"的位置——第 12 步会改进这里。

#### 6.2.4 作用域的四个函数

```c
static void env_init(Env *e, Env *parent) {
    e->vars   = NULL;
    e->count  = 0;
    e->cap    = 0;
    e->parent = parent;
}

static void env_free(Env *e) {
    free(e->vars);
    e->vars  = NULL;
    e->count = 0;
    e->cap   = 0;
}
```

**逐行解释**：`env_init` 把 `vars` 置成 `NULL` 是必须的——
`free(NULL)` 是合法的（等于什么都不做），但如果 `env_free` 被调用时 `vars`
还是野指针，就会崩。

**内存管理的约定**：`env_init` / `env_free` 必须配对。
本步有三处 `env_init`（代码块、全局环境），所以也有三处 `env_free`。
第 10 步函数调用会再增加一处——**每增加一个 `Env`，就要记得在所有出口路径上 `env_free`**。

```c
static int find_var(Env *e, const char *name) {
    for (int i = e->count - 1; i >= 0; i--)
        if (!strcmp(e->vars[i].name, name)) return i;
    return -1;
}
```

**逐行解释**：`i` 从 `count - 1` 倒着走到 0。`count` 是 0 时，
`i` 从 -1 开始，循环条件 `i >= 0` 直接为假，返回 -1——不需要特判空表。

#### 6.2.5 `Flow`：提前定型的控制流信号

```c
typedef struct {
    int   returned;  /* 1 表示这条语句链里有 return  */
    Value value;     /* return 带回来的值          */
} Flow;

static Flow noflow(void) {
    Flow f;
    f.returned = 0;
    f.value = number(0);
    return f;
}
```

**逐行解释**：`noflow()` 是"什么都没发生"的快捷方式。
现在 `returned` 永远是 0，但**字段先占好位置**——第 11 步只要改 `exec` 的一个 `case`，
不用改任何签名。

**如果你写错了会怎样**：如果第 5 步让 `exec` 返回 `void`，
第 11 步加 `return` 时就得把 `exec` 的签名和 7 个调用点全改一遍——
那种"全文件级"的改动会污染 diff，读者根本看不出第 11 步到底加了什么。

#### 6.2.6 `exec()`：执行语句

```c
static Flow exec(Node *n, Env *e) {
    if (!n) return noflow();

    Flow f = noflow();
    switch (n->type) {
        case N_BLOCK: {
            Env local;
            env_init(&local, e);
            for (int i = 0; i < n->count; i++) {
                f = exec(n->items[i], &local);
                if (f.returned) { env_free(&local); return f; }
            }
            env_free(&local);
            return noflow();
        }
        ...
```

**逐行解释**：

- `if (!n) return noflow();` —— 允许传入 `NULL`。第 6 步的 `if` 没有 `else` 时
  就会走到这里（"条件为假，什么都不做"）。
- 代码块**新建一个 `Env`**，父环境是 `e`。这就是块级作用域的来源。
- `if (f.returned) { env_free(&local); return f; }` —— 一旦有 `return`，
  **立刻释放作用域并把结果往上传**。注意这里是 `env_free` **然后** `return`，
  顺序不能反（先 return 就泄漏了）。

**如果你写错了会怎样**：漏掉 `env_free(&local)` 会**内存泄漏**——
每个代码块泄漏一次变量表的堆内存。测试用例多跑几次可能看不出来，
但程序跑久了就会吃掉大量内存。用 `valgrind ./minilang demo.ml` 能查出来。

#### 6.2.7 `eval()` 的三处改动

```diff
         case N_VAR:
-            fatal_at(n->tok, "求值器还不支持变量，变量会在讲作用域时支持");
-            break;
+            /* 变量引用：沿着作用域链找 */
+            return number(get_var(e, n->tok, n->text));
```

```diff
         case N_CALL:
-            fatal_at(n->tok, "求值器还不支持函数调用，内置函数会在后面一步支持");
-            break;
+        case N_CALL: {
+            /* 目前只认识 print 一个内建函数。
+               第 8 步会把它和 abs/sqrt 等一起收进一张"内置函数表"。 */
+            if (strcmp(n->text, "print") != 0)
+                fatal_at(n->tok, "未知函数 '%s'", n->text);
+            ...
+        }
```

**逐行解释**：

- `N_VAR` 从"报错"变成"查作用域"，这是本步最核心的一行。
- `N_CALL` 从 `case` 变成 `case ... {`，因为里面要声明局部变量
  （`Value v = eval(...)`），C 语言要求**带语句块的 case 必须加花括号**，
  否则变量会泄漏到后面的 case 里（编译器会报错）。
- `print` 是本步唯一的内建函数，而且**直接写在 `eval` 里**——
  第 8 步会把它挪进一张正式的"内置函数表"。
- `printf("%.15g", v.value)` —— 数字之间不会自动换行，这点在第 9 步会改善。

```diff
+        default:
+            /* 语句节点不能出现在表达式里 */
+            fatal_at(n->tok, "此处不是表达式");
```

**逐行解释**：`eval` 的 `switch` 之前只处理表达式节点，现在 `NodeType` 里多了四种
语句节点，编译器会警告"switch 没有处理全部枚举值"（`-Wswitch`）。
加一个 `default` 分支既消掉了警告，也给出准确的错误信息。

#### 6.2.8 两种模式都改成"整个程序"

```diff
 static int mode_ast(const char *path) {
     src = read_file(path);
     lex();
-    Node *e = expression();
-    if (cur()->type != T_EOF)
-        fatal_at(*cur(), "语法错误：表达式结束后还有多余的 '%s'", cur()->text);
+    Node *prog = parse_program();
     printf("语法树：\n");
-    dump_ast(e, "");
-    Value v = eval(e, NULL);
-    printf("求值结果：%g\n", v.value);
+    dump_ast(prog, "");
     free((void *)src);
     return had_error ? 1 : 0;
 }
```

**逐行解释**：`mode_ast` 从"解析一个表达式"变成"解析整个程序"。
注意**求值那两行被删掉了**——因为程序里的语句没有"一个总的值"可打印。
想看求值结果，请用 `run` 模式。

```diff
--- a/minilang.c
+++ b/minilang.c
@@ -31,6 +31,8 @@
 #define MAX_NODES  20000
 #define MAX_TEXT   4096
 #define MAX_PARAMS 32
+#define MAX_VARS   2048
+#define MAX_NAME   128
 
 /* ------------------------------------------------------------------ */
 /* Token：词法分析器产出的最小单位                                     */
@@ -265,7 +267,11 @@
     N_VAR,      /* 变量引用          */
     N_UNARY,    /* 一元运算：-x !x   */
     N_BINARY,   /* 二元运算：a + b   */
-    N_CALL      /* 函数调用：f(a,b)  */
+    N_CALL,     /* 函数调用：f(a,b)  */
+    N_BLOCK,    /* 代码块 { ... }    */
+    N_LET,      /* let x = 1;        */
+    N_ASSIGN,   /* x = 1;            */
+    N_EXPR      /* 表达式语句        */
 } NodeType;
 
 typedef struct Node Node;
@@ -421,6 +427,71 @@
 
 /* 表达式的最外层入口 */
 static Node *expression(void) { return logical_or(); }
+
+/* ------------------------------------------------------------------ */
+/* 语法分析器：语句                                                    */
+/* ------------------------------------------------------------------ */
+static Node *statement(void);
+
+/* 代码块：{ 语句 语句 ... } */
+static Node *block(void) {
+    Token t = expect(T_LB, "代码块需要 '{'");
+    Node *n = node(N_BLOCK, t);
+    while (cur()->type != T_RB && cur()->type != T_EOF) {
+        if (n->count >= 128) fatal_at(*cur(), "单个代码块语句过多（上限 128）");
+        n->items[n->count++] = statement();
+    }
+    expect(T_RB, "代码块缺少 '}'");
+    return n;
+}
+
+/* 一条语句 */
+static Node *statement(void) {
+    Token t = *cur();
+
+    /* 裸代码块 { ... } */
+    if (cur()->type == T_LB) return block();
+
+    /* let 变量 = 表达式; */
+    if (match(T_LET)) {
+        Node *n = node(N_LET, t);
+        Token id = expect(T_ID, "let 后需要变量名");
+        snprintf(n->text, sizeof(n->text), "%s", id.text);
+        expect(T_ASSIGN, "变量定义需要 '='");
+        n->a = expression();
+        expect(T_SEMI, "语句末尾需要 ';'");
+        return n;
+    }
+
+    /* 赋值语句：标识符 = 表达式;
+       这里要特别小心不要把 == 当成 =：
+       靠"当前是标识符，且下一个 token 是 T_ASSIGN"来判断。 */
+    if (cur()->type == T_ID && toks[pos + 1].type == T_ASSIGN) {
+        Token id = *cur();
+        pos += 2;
+        Node *n = node(N_ASSIGN, id);
+        snprintf(n->text, sizeof(n->text), "%s", id.text);
+        n->a = expression();
+        expect(T_SEMI, "赋值语句末尾需要 ';'");
+        return n;
+    }
+
+    /* 其它都是表达式语句：表达式; */
+    Node *n = node(N_EXPR, t);
+    n->a = expression();
+    expect(T_SEMI, "语句末尾需要 ';'");
+    return n;
+}
+
+/* 整个程序：一串语句 */
+static Node *parse_program(void) {
+    Node *n = node(N_BLOCK, toks[0]);
+    while (cur()->type != T_EOF) {
+        if (n->count >= 128) fatal_at(*cur(), "顶层语句过多（上限 128）");
+        n->items[n->count++] = statement();
+    }
+    return n;
+}
 
 /* ------------------------------------------------------------------ */
 /* 打印语法树                                                          */
@@ -456,6 +527,22 @@
             printf("调用 %s\n", n->text);
             for (int i = 0; i < n->count; i++) kids[m++] = n->items[i];
             break;
+        case N_BLOCK:
+            printf("代码块\n");
+            for (int i = 0; i < n->count; i++) kids[m++] = n->items[i];
+            break;
+        case N_LET:
+            printf("定义 %s\n", n->text);
+            kids[m++] = n->a;
+            break;
+        case N_ASSIGN:
+            printf("赋值 %s\n", n->text);
+            kids[m++] = n->a;
+            break;
+        case N_EXPR:
+            printf("表达式语句\n");
+            kids[m++] = n->a;
+            break;
     }
 
     for (int i = 0; i < m; i++) {
@@ -471,10 +558,87 @@
 /* 运行时：值与求值                                                    */
 /* ------------------------------------------------------------------ */
 
-/* 环境（作用域）。这一步还用不上它，但先把参数位置留出来：
-   第 5 步讲作用域时就不用改 eval 的签名，也不用动所有调用点。 */
-typedef struct Env Env;
+/* 一个变量 */
+typedef struct {
+    char   name[MAX_NAME];
+    double value;
+} Var;
+
+/* 环境（作用域）：一串变量 + 指向外层作用域的指针。
+   第 3 步只是留了个空壳，现在把它真正填上。 */
+struct Env {
+    Var   *vars;        /* 变量表，按需增长（不是固定长度数组！） */
+    int    count;       /* 已经定义了多少个变量 */
+    int    cap;         /* 变量表当前能放多少个 */
+    struct Env *parent; /* 外层作用域，NULL 表示最外层 */
+};
+typedef struct Env Env;   /* 后面就可以直接写 Env 而不是 struct Env */
+
+/* 作用域的建立与销毁 */
+static void env_init(Env *e, Env *parent) {
+    e->vars   = NULL;
+    e->count  = 0;
+    e->cap    = 0;
+    e->parent = parent;
+}
+
+static void env_free(Env *e) {
+    free(e->vars);
+    e->vars  = NULL;
+    e->count = 0;
+    e->cap   = 0;
+}
+
+/* 在当前作用域里加一个变量。变量表是动态数组：
+   用多少占多少，不用一开始就开 2048 个槽位。 */
+static void env_push(Env *e, const char *name, double v) {
+    if (e->count >= MAX_VARS)
+        fatal_at((Token){.line = 0, .col = 0}, "变量数量超出上限");
+    if (e->count == e->cap) {
+        int ncap = e->cap ? e->cap * 2 : 8;
+        Var *nv = (Var *)realloc(e->vars, (size_t)ncap * sizeof(Var));
+        if (!nv) { perror("realloc"); exit(1); }
+        e->vars = nv;
+        e->cap  = ncap;
+    }
+    snprintf(e->vars[e->count].name, sizeof(e->vars[e->count].name), "%s", name);
+    e->vars[e->count].value = v;
+    e->count++;
+}
+
+/* 在当前作用域里找变量。倒着找：后定义的遮蔽先定义的。 */
+static int find_var(Env *e, const char *name) {
+    for (int i = e->count - 1; i >= 0; i--)
+        if (!strcmp(e->vars[i].name, name)) return i;
+    return -1;
+}
 
+/* 沿着作用域链往外找。找不到就是"未定义变量"。 */
+static double get_var(Env *e, Token t, const char *name) {
+    for (Env *p = e; p; p = p->parent) {
+        int i = find_var(p, name);
+        if (i >= 0) return p->vars[i].value;
+    }
+    fatal_at(t, "未定义变量 '%s'", name);
+    return 0;
+}
+
+/* 定义变量：只能在当前作用域，同名就是重复定义 */
+static void define_var(Env *e, Token t, const char *name, double v) {
+    if (find_var(e, name) >= 0)
+        fatal_at(t, "变量 '%s' 在当前作用域中已定义", name);
+    env_push(e, name, v);
+}
+
+/* 赋值：可以改外层作用域的变量 */
+static void set_var(Env *e, Token t, const char *name, double v) {
+    for (Env *p = e; p; p = p->parent) {
+        int i = find_var(p, name);
+        if (i >= 0) { p->vars[i].value = v; return; }
+    }
+    fatal_at(t, "不能给未定义变量 '%s' 赋值", name);
+}
+
 /* 一个值。MiniLang 里普通变量只能是数字，字符串只出现在 print 里，
    所以 Value 用一个标记位区分两者。
    is_string 和 string 这两个字段现在还没用上，是提前留好的。 */
@@ -493,11 +657,25 @@
     return v;
 }
 
+/* 控制流信号：一条语句执行完可能什么都不返回，也可能提前"return"。
+   returned 字段从现在起就定型了，后面加 return 语句时不用改 exec 的签名。 */
+typedef struct {
+    int   returned;  /* 1 表示这条语句链里有 return  */
+    Value value;     /* return 带回来的值          */
+} Flow;
+
+/* 什么都不返回 */
+static Flow noflow(void) {
+    Flow f;
+    f.returned = 0;
+    f.value = number(0);
+    return f;
+}
+
 /* 求值：把一棵表达式树算成一个值。
    递归的结构和语法分析器正好相反：分析时"从左到右展开"，
    求值时"从叶子往回算"。 */
 static Value eval(Node *n, Env *e) {
-    (void)e;   /* 目前还用不到环境，第 5 步开始会真正用到它 */
     if (!n) return number(0);
 
     switch (n->type) {
@@ -509,12 +687,22 @@
             break;
 
         case N_VAR:
-            fatal_at(n->tok, "求值器还不支持变量，变量会在讲作用域时支持");
-            break;
+            /* 变量引用：沿着作用域链找 */
+            return number(get_var(e, n->tok, n->text));
 
-        case N_CALL:
-            fatal_at(n->tok, "求值器还不支持函数调用，内置函数会在后面一步支持");
-            break;
+        case N_CALL: {
+            /* 目前只认识 print 一个内建函数。
+               第 8 步会把它和 abs/sqrt 等一起收进一张"内置函数表"。 */
+            if (strcmp(n->text, "print") != 0)
+                fatal_at(n->tok, "未知函数 '%s'", n->text);
+            if (n->count != 1)
+                fatal_at(n->tok, "函数 'print' 参数数量错误：需要 1 个，实际 %d 个", n->count);
+            Value v = eval(n->items[0], e);
+            if (v.is_string)
+                fatal_at(n->tok, "print 的参数必须是数字");
+            printf("%.15g", v.value);   /* 数字之间不会自动换行 */
+            return number(0);
+        }
 
         case N_UNARY: {
             Value a = eval(n->a, e);
@@ -559,10 +747,56 @@
             }
             break;
         }
+
+        default:
+            /* 语句节点不能出现在表达式里 */
+            fatal_at(n->tok, "此处不是表达式");
     }
     return number(0);
 }
 
+/* 执行一条语句。返回值是一个 Flow，告诉调用者"要不要带着结果往上传"。 */
+static Flow exec(Node *n, Env *e) {
+    if (!n) return noflow();
+
+    Flow f = noflow();
+    switch (n->type) {
+        case N_BLOCK: {
+            /* 代码块自带一个作用域：里面的 let 不影响外面 */
+            Env local;
+            env_init(&local, e);
+            for (int i = 0; i < n->count; i++) {
+                f = exec(n->items[i], &local);
+                if (f.returned) { env_free(&local); return f; }
+            }
+            env_free(&local);
+            return noflow();
+        }
+
+        case N_LET: {
+            Value v = eval(n->a, e);
+            if (v.is_string) fatal_at(n->tok, "变量目前只支持数字类型");
+            define_var(e, n->tok, n->text, v.value);
+            return f;
+        }
+
+        case N_ASSIGN: {
+            Value v = eval(n->a, e);
+            if (v.is_string) fatal_at(n->tok, "变量目前只支持数字类型");
+            set_var(e, n->tok, n->text, v.value);
+            return f;
+        }
+
+        case N_EXPR:
+            (void)eval(n->a, e);
+            return f;
+
+        default:
+            fatal_at(n->tok, "这里不是一条语句");
+    }
+    return f;
+}
+
 /* ------------------------------------------------------------------ */
 /* 读文件                                                              */
 /* ------------------------------------------------------------------ */
@@ -615,21 +849,29 @@
 static int mode_ast(const char *path) {
     src = read_file(path);
     lex();
-    Node *e = expression();
-    if (cur()->type != T_EOF)
-        fatal_at(*cur(), "语法错误：表达式结束后还有多余的 '%s'", cur()->text);
+    Node *prog = parse_program();
     printf("语法树：\n");
-    dump_ast(e, "");
-    Value v = eval(e, NULL);
-    printf("求值结果：%g\n", v.value);
+    dump_ast(prog, "");
     free((void *)src);
     return had_error ? 1 : 0;
 }
 
 static int mode_run(const char *path) {
-    (void)path;
-    fprintf(stderr, "解释执行尚未实现，请先用 tokens 或 ast 模式查看结果\n");
-    return 1;
+    src = read_file(path);
+    lex();
+    Node *prog = parse_program();
+
+    Env global;
+    env_init(&global, NULL);
+    for (int i = 0; i < prog->count; i++) {
+        Flow f = exec(prog->items[i], &global);
+        if (f.returned)
+            fatal_at(prog->items[i]->tok, "return 不能出现在函数外部");
+    }
+    env_free(&global);
+
+    free((void *)src);
+    return had_error ? 1 : 0;
 }
 
 int main(int argc, char **argv) {
```

**逐行解释**：

- 全局环境 `Env global` 的 `parent` 是 `NULL`，表示"最外层"。
- `exec(prog->items[i], &global)` —— **顶层语句直接跑在全局环境里**，
  所以顶层的 `let x = 1;` 定义的是全局变量。
- `if (f.returned)` 现在还不可能为真（第 11 步才有 `return`），
  但**先写好**，第 11 步就不用改这里了。
- `env_free(&global)` —— 程序跑完释放全局变量表。

### 6.3 完整 diff

> **怎么用这段 diff**：推荐**照着敲**——每改一小段就编译一次，比一次性粘贴更容易记住。
> 如果想直接应用，把下面的内容存成 `step.patch`，然后在**上一步的目录**里执行：
>
> ```sh
> patch -p1 < ../005/step.patch
> ```
>
> 应用完和本目录的 `minilang.c` **逐字节一致**（可以 `diff` 验证）。

<details>
<summary>点开看完整 diff（真实 <code>diff -u</code> 输出）</summary>

```diff
```

</details>

### 6.4 插入位置说明

| 代码块 | 放在哪里 |
|---|---|
| `MAX_VARS` / `MAX_NAME` 宏 | 现有 `#define` 区，`MAX_PARAMS` 之后 |
| `N_BLOCK` 等四个枚举值 | `NodeType` 枚举里，`N_CALL` 之后 |
| `statement()` / `block()` / `parse_program()` | `expression()` 之后、"打印语法树"注释块之前 |
| `dump_ast()` 的四个新 `case` | `dump_ast()` 的 `switch` 里，`case N_CALL:` 之后 |
| `Var` / `Env` 结构体、`env_*` / `find_var` / `get_var` / `define_var` / `set_var` | "运行时：值与求值"区里，`Value` 结构体**之前** |
| `Flow` / `noflow()` | `number()` 之后、`eval()` 之前 |
| `exec()` | `eval()` 之后、"读文件"之前 |
| `mode_ast` / `mode_run` 的改动 | 原位修改 |

## 7. 测试

`./run.sh` 的真实输出：

```
▶ 用例（模式：run）
  ✅ case_01_let_and_assign.ml    [run]
  ✅ case_02_block_scope.ml       [run]
  ✅ case_03_shadow.ml            [run]
  ✅ case_04_undefined_var.ml     [run]
  ✅ case_05_redefine.ml          [run]
  ✅ case_06_assign_undefined.ml  [run]
▶ demo.ml
  | 10015
✅ 全部通过（7 项）
```

逐个用例说明：

### case_01：定义 + 赋值

```
let x = 10;
x = x + 5;
print(x);
```

输出 `15`。`x = x + 5` 走的是 `set_var`：先算出右边 `15`，
再沿着作用域链找到 `x` 改掉它的值。

### case_02：块作用域

```
let x = 1;
{
    let x = 2;
    print(x);
}
print(x);
```

输出 `21`——块里打印 `2`（内层遮蔽外层），块外打印 `1`。
块结束时 `env_free(&local)` 释放了内层作用域，外层的 `x` 完好无损。

### case_03：遮蔽后外层不受影响

```
let x = 1;
{
    let x = 2;
}
print(x);
```

输出 `1`。内层的 `x` 在块结束时被释放，外层的 `x` 还是 1。

### case_04：未定义变量（错误用例）

```
let x = price + 1;
```

期望 stderr：

```
错误 [1:9]: 未定义变量 'price'
```

期望退出码 `1`。注意**报错位置是 `price` 那个 token 的位置**（第 1 行第 9 列），
不是整条语句的位置——这是因为 `get_var` 收到的是 `N_VAR` 节点自己的 token。

### case_05：同作用域重复定义（错误用例）

```
let x = 1;
let x = 2;
```

期望 stderr：

```
错误 [2:1]: 变量 'x' 在当前作用域中已定义
```

期望退出码 `1`。位置是**第二条** `let` 的位置。

### case_06：给未定义变量赋值（错误用例）

```
y = 5;
```

期望 stderr：

```
错误 [1:1]: 不能给未定义变量 'y' 赋值
```

期望退出码 `1`。

### 自己动手试

```sh
cd 005
./minilang demo.ml ast      # 看整个程序的语法树
```

```
语法树：
代码块
├── 定义 x
│   └── 数字 10
├── 赋值 x
│   └── 二元 PLUS
│       ├── 变量 x
│       └── 数字 5
├── 代码块
│   ├── 定义 x
│   │   └── 数字 100
│   └── 表达式语句
│       └── 调用 print
│           └── 变量 x
└── 表达式语句
    └── 变量 x
```

## 8. 易错点与坑

### 坑 1：`env_init` 和 `env_free` 忘了配对

**现象**：程序跑久了内存暴涨；`valgrind ./minilang demo.ml` 报一堆 "definitely lost"。

**原因**：`Env` 里的 `vars` 是 `realloc` 出来的堆内存，每次进出一个代码块
都要 `env_free` 一次。

**正确写法**：`env_init` 之后，**每一条返回路径**都要 `env_free`：

```c
for (int i = 0; i < n->count; i++) {
    f = exec(n->items[i], &local);
    if (f.returned) { env_free(&local); return f; }   /* ← 提前返回也要释放！ */
}
env_free(&local);
```

### 坑 2：`realloc` 的返回值必须检查

**现象**：内存紧张时程序直接崩，或者更糟——原来的指针被覆盖，内存泄漏。

**原因**：`realloc` 失败时返回 `NULL`，**原来的那块内存还在**（没被释放，也没被搬走）。

**正确写法**：

```c
Var *nv = (Var *)realloc(e->vars, (size_t)ncap * sizeof(Var));
if (!nv) { perror("realloc"); exit(1); }
e->vars = nv;          /* 只有成功了才能覆盖旧指针 */
```

写成 `e->vars = realloc(e->vars, ...)` 的话，失败时 `e->vars` 变成 `NULL`，
旧内存再也找不回来了。

### 坑 3：忘了 `realloc` 可能搬动内存

**现象**：`env_push` 里保存了 `e->vars[i]` 的地址，之后 `realloc` 之后那个地址失效。

**正确写法**：**永远不要保存 `vars` 数组元素的地址**。要用的时候用 `e->vars[i]` 重新取。

### 坑 4：`switch` 的 `case` 里声明变量忘了加花括号

**现象**：编译报

```
error: declaration of 'v' shadows a previous local
```

或者更隐蔽的——变量在别的 `case` 里也可见。

**原因**：C 的 `case` 只是标签，不是作用域。`case N_CALL: { ... }` 里的花括号
才是真正的作用域。

**正确写法**：只要 `case` 里有变量声明，就加花括号。

### 坑 5：把"遮蔽"当成"修改"

**现象**：写

```
let x = 1;
{ let x = 2; }
print(x);        /* 期望 2，实际 1 */
```

**原因**：块里的 `let` 是**新建**一个变量，不是修改外层的。`define_var` 只看当前作用域。

**正确写法**：想改外层的值，用赋值 `x = 2;`（`set_var` 会往外找）。

### 坑 6：`find_var` 正着找

**现象**：函数参数和局部变量重名时，取到了错的那个。

**原因**：变量表是数组，"后定义的应该赢"。

**正确写法**：`for (int i = e->count - 1; i >= 0; i--)`——从后往前找。

### 坑 7：`case` 里漏了 `break` 引发"隐式贯穿"

**现象**：加了 `N_LET` 之后，编译器报

```
warning: implicit fallthrough from switch statement
```

**原因**：C 的 `switch` 不会自动结束一个 `case`，会**接着往下执行**下一个 `case`。

**正确写法**：每个 `case` 结尾要么 `break`/`return`，要么干脆写成 `case ...: { ... return ...; }`。

## 9. 自测题

### 9.1 概念题

**Q1.** `let` 和赋值在作用域上有什么区别？

<details><summary>参考答案</summary>

`let`（`define_var`）**只在当前作用域**新建变量，同名会报"已定义"；
赋值（`set_var`）**沿作用域链往外找**，找到第一个同名变量就改它的值，
找不到就报"不能给未定义变量赋值"。

</details>

**Q2.** 为什么 `get_var` 要沿 `parent` 链往外找，而 `define_var` 只看当前层？

<details><summary>参考答案</summary>

因为"读一个变量"应该能看到外层（词法作用域的规则），
而"定义一个变量"必须只影响当前作用域，否则块里的 `let` 会污染全局。
同一个作用域重复 `let` 报错，但内层 `let` 遮蔽外层是允许的。

</details>

**Q3.** `sizeof(Env)` 在固定数组版和动态数组版分别是多少？差多少倍？

<details><summary>参考答案</summary>

固定数组版（`Var vars[2048]`）：`2048 × 136 + 16` ≈ **272 KB**；
动态数组版（`Var *vars`）：`8 + 4 + 4 + 8` = **24 字节**。
差约 **11000 倍**。这个差距在递归时会直接决定"能递归多少层"。

</details>

### 9.2 预测题（改一行会发生什么）

**Q4.** 把 `find_var` 里的 `i = e->count - 1` 改成 `i = 0`，本步的测试会挂吗？

<details><summary>参考答案</summary>

**不会挂。** 因为本步的 `define_var` 禁止同作用域重复定义，
所以同一个作用域里永远不会有两个同名变量，正着找和倒着找结果一样。
这个 bug 要到第 10 步（函数参数和局部变量重名）才会暴露。

</details>

**Q5.** 把 `exec` 里 `case N_BLOCK` 的 `env_init(&local, e)` 改成 `env_init(&local, NULL)`，会发生什么？

<details><summary>参考答案</summary>

块里的变量**再也找不到外层的变量**了。
`{ let x = 2; }` 里的 `x` 和外层的 `x` 彻底断开关系，
块里如果要用外层的变量会报"未定义变量"。
`case_02_block_scope.ml` 会挂（输出变成 `2` + 报未定义变量，而不是 `21`）。

</details>

**Q6.** 如果把 `case N_ASSIGN` 里的 `set_var` 换成 `define_var`，`case_01` 会输出什么？

<details><summary>参考答案</summary>

`x = x + 5;` 会报"变量 'x' 在当前作用域中已定义"，
因为 `define_var` 不允许重复定义。程序在第二行就中止，退出码 1。

</details>

### 9.3 动手题

**Q7.** 在 `005/` 下新建 `my_cases/`，写三个 `.ml` 文件并用 `./minilang` 跑出真实输出：

```
t1.ml:  let x = 1; { let x = 2; } print(x);        → 1
t2.ml:  let x = 1; { x = 2; } print(x);            → 2
t3.ml:  let x = 1; print(x + 1);                   → 2
```

做完想一想：t1 和 t2 的**语法**都是合法的，区别只在 `let` 和 `=` 的语义。

<details><summary>参考答案</summary>

t1 输出 `1`（块里的 `let` 是新变量，块外看不到）；
t2 输出 `2`（`=` 是赋值，沿链找到外层的 `x` 改了它的值）；
t3 输出 `2`。

</details>

**Q8.** 给 `005/minilang.c` 加一个 `env_dump(Env *e)` 函数，把当前作用域里所有变量名和值打印出来，在 `case N_LET` 定义完之后调用一次。看看输出和你想的是否一致。

<details><summary>参考答案</summary>

参考实现：

```c
static void env_dump(Env *e, int depth) {
    for (int i = 0; i < depth; i++) printf("  ");
    printf("作用域 %p：%d 个变量\n", (void *)e, e->count);
    for (int i = 0; i < e->count; i++) {
        for (int k = 0; k < depth; k++) printf("  ");
        printf("  %s = %g\n", e->vars[i].name, e->vars[i].value);
    }
    if (e->parent) env_dump(e->parent, depth + 1);
}
```

注意打印变量表时**正着遍历**（想按定义顺序看），而查找时是倒着遍历。

</details>

**Q9.** 把 `005/minilang.c` 里 `env_push` 的扩容步长从 `× 2` 改成 `× 8`，跑 `./run.sh`，结果有变化吗？

<details><summary>参考答案</summary>

测试结果不变（行为完全一样），只是内存占用策略变了：
`× 2` 更省内存，`× 8` 扩容次数更少、拷贝总次数更少。
第 5 步的用例里每个作用域只有 1~2 个变量，两种策略都只扩容一次。

</details>

### 9.4 延伸题

**Q10.** 想让 `let` 允许在同一个作用域里重复定义（后定义的赢），应该改哪个函数？

<details><summary>参考答案</summary>

改 `define_var`：去掉"已定义就报错"的检查即可。
但这样一来 `find_var` 倒着找就**真的重要**了（否则取到的是第一个），
而且 `case_05_redefine.ml` 这个测试用例的期望输出要改。
这说明"禁止重复定义"和"倒着查找"是配套的设计。

</details>

## 10. 本步的设计取舍

- **动态数组 vs 固定数组**：本步选了动态数组。当时的理由是"省内存"，
  真正的理由是**递归**（第 11 步会证明）。回头看，这是一处"为未来买单"的设计，
  代价是 `env_push` 里多了 `realloc` 的样板代码。
- **`eval` / `exec` 分离**：代价是同一个 AST 节点有时走 `eval` 有时走 `exec`，
  换来的是"表达式有值、语句没值"这个语义差别清清楚楚。
- **`Flow` 提前定型**：本步的 `returned` 字段一次都没被用过，
  纯粹是为了第 11 步不用改签名。代价是读者第一次看到 `Flow` 时会觉得"这有啥用"。
- **顶层语句上限 128**：这个数字有点小。教学版先够用，第 12 步会把这张上限表列全。

## 11. 下一步预告

现在程序能跑起来了，但还**不会分支**——写什么就执行什么，永远按顺序走。

下一节：[第 6 步 · if / else](../006/README.md)，我们会加上：

- `if (条件) 语句` 和 `if (条件) 语句 else 语句`；
- 一个新的 AST 节点 `N_IF`，用 `a`/`b`/`c` 三个子节点分别装条件、then 分支、else 分支；
- **没有 `else` 的时候 `n->c` 是 `NULL`**——这正好用到本步 `memset` 的功劳，
  以及 `exec()` 开头那句 `if (!n) return noflow();`。
