# 第 10 步 · 用户函数：定义与调用

> 进度 █████████░ 10/13 ｜ 上一节：[第 9 步 · print 与字符串模板](../009/README.md) ｜ 下一节：[第 11 步 · return 与递归](../011/README.md)

## 1. 本步目标

读完你将能够：

- 写出一个函数定义 `fn add(a, b) { ... }` 并调用它。
- 理解"**先注册再执行**"：为什么函数表必须在执行任何语句之前填好。
- 解释每次调用为什么都要**开一个新的作用域**，参数就是作用域里的变量。
- 理解函数里的 `let` 和 `=` 的区别（前者新建局部变量，后者改外层的）。
- 用 `call_depth` 计数器挡住无限递归。
- 知道本步函数**还没有返回值**——`return` 是下一步的内容。

## 2. 运行方式

```sh
cd 010
./run.sh                 # 编译 + 跑全部用例 + 跑 demo
./minilang demo.ml       # 默认 run 模式
./minilang demo.ml ast   # 看函数定义的语法树
```

本步 `run.sh` 的真实输出见 [第 7 节](#7-测试)。

## 3. 现在的程序能做什么

前九步已经能写"一个程序"了，但函数只能调用**内置函数**。
从这一步起，你可以在 MiniLang 里定义自己的函数。

| 能力 | 例子 | 状态 |
|---|---|---|
| 内置函数 | `abs(-3)`、`sqrt(16)` | 第 8 步就有 |
| 用户函数定义 | `fn add(a, b) { ... }` | **本步新增** |
| 用户函数调用 | `add(3, 4)` | **本步新增** |
| 函数返回值 | `return 1;` | 还没有，见第 11 步 |

`010/demo.ml` 的内容：

```
// 第 10 步：自己定义函数
fn add(a, b) {
    print(a + b);
}
add(3, 4);
```

真实运行：

```sh
$ ./minilang demo.ml
7
```

注意 `print` 依然不换行，所以输出就是光秃秃的 `7`。

**本步的一个"半成品"状态**：函数**没有返回值**。
`let x = add(3, 4);` 里的 `x` 会是 `0`，因为函数体没有 `return`。
这是刻意的——`return` 需要第 5 步定型的 `Flow` 结构配合，第 11 步再补。

## 4. 前置知识回顾

第 5 步我们建立了作用域模型：

```text
        ┌───────────────┐
        │ global        │  parent = NULL
        └───────┬───────┘
                │ parent
        ┌───────▼───────┐
        │ block          │
        └───────────────┘
```

本步要问的问题是：**函数调用时，这个链条长什么样？**

答案：函数体是一个代码块，所以它天然会得到一个新的 `Env`：

```text
        ┌───────────────┐
        │ global        │
        └───────┬───────┘
                │ parent
        ┌───────▼───────┐
        │ 函数局部作用域   │  ← 每次调用都新建一个
        │  a = 3, b = 4  │  ← 参数就是这里面的变量
        └───────┬───────┘
                │ parent
        ┌───────▼───────┐
        │ 函数体的代码块   │  ← 函数体本身是 block()，又多一层
        └───────────────┘
```

## 5. 本步引入的核心概念

### 5.1 概念一：函数声明长什么样

```
fn 名字(参数, 参数, ...) { 函数体 }
```

对应到 AST：

| 节点 | 存什么 |
|---|---|
| `N_FN` | `text` = 函数名，`params[]` = 参数名，`a` = 函数体（一个 `N_BLOCK`） |

```c
if (match(T_FN)) {
    Node *n = node(N_FN, t);
    Token id = expect(T_ID, "fn 后需要函数名");
    snprintf(n->text, sizeof(n->text), "%s", id.text);
    expect(T_LP, "函数参数列表需要 '('");
    if (cur()->type != T_RP) {
        do {
            if (n->nparams >= MAX_PARAMS)
                fatal_at(*cur(), "函数最多支持 %d 个参数", MAX_PARAMS);
            Token p = expect(T_ID, "参数必须是变量名");
            n->params[n->nparams++] = dupstr(p.text);   /* 参数名要留到运行时用，复制一份 */
        } while (match(T_COMMA));
    }
    expect(T_RP, "参数列表需要 ')'");
    n->a = block();      /* 函数体也是一个代码块 */
    return n;
}
```

**代码要点**：

- `do { ... } while (match(T_COMMA))` 是处理逗号分隔列表的惯用法：
  先处理第一个，然后只要后面还有逗号就继续。
- `if (cur()->type != T_RP)` 让**零参数函数** `fn f() { ... }` 也能写。
- `n->a = block()` —— 函数体复用第 5 步的代码块解析器，
  所以函数体天然就是一个新作用域，不需要额外处理。

### 5.2 概念二：参数名为什么要 `dupstr` 复制一份

```c
n->params[n->nparams++] = dupstr(p.text);
```

`p.text` 是什么？它是 `Token` 结构体里的一个**定长字符数组**：

```c
typedef struct {
    TokType type;
    char    text[MAX_TEXT];   /* ← 就在 toks[] 数组里 */
    double  num;
    int     line, col;
} Token;
```

参数名要保存到**运行期**（调用时才知道实参的值，那时要把名字放进新作用域的变量表）。
如果直接 `n->params[i] = p.text`，`params` 的类型就得是 `char (*)[MAX_TEXT]`——
一个"指向数组的指针"，用起来很别扭。

更根本的问题是**生命周期**：`toks[]` 是一个全局数组，程序结束才释放，
所以理论上指着它并不会立刻出问题。但：

1. `params` 是 `char *`，存一个指针最自然；
2. 复制一份之后，参数名就**不再依赖 token 数组**，将来换掉词法分析器的存储方式
   （比如改成动态分配）时，这里一行都不用改。

```c
/* 复制一份字符串（参数名要保存到运行期，所以必须复制，不能指着源码） */
static char *dupstr(const char *s) {
    size_t n = strlen(s) + 1;
    char *p = (char *)malloc(n);
    if (!p) { perror("malloc"); exit(1); }
    memcpy(p, s, n);
    return p;
}
```

**如果你写错了会怎样**：写成 `n->params[n->nparams++] = p.text;` 编译不过
（`char[MAX_TEXT]` 不能赋给 `char *`）。如果把 `params` 的类型改成
`char params[MAX_PARAMS][MAX_NAME]`，能编译，但每个节点要多占 `32 × 128 = 4 KB`——
20000 个节点就是 80 MB。

**代价**：这些 `dupstr` 分配的内存** never freed**（函数参数表活到程序结束）。
教学版接受这个泄漏；正式实现应该用 arena（内存池）统一回收。

### 5.3 概念三：函数表与"先注册再执行"

```c
typedef struct {
    char  name[MAX_NAME];  /* 函数名     */
    Node *decl;            /* 函数声明节点（名字、参数、函数体都在里面） */
} Func;

static Func funcs[MAX_FUNCS];
static int  nfuncs;
```

`Func` 里只存了**名字**和**声明节点的指针**。参数名、函数体都在 `decl` 指向的
`N_FN` 节点里——不重复存。

登记函数：

```c
static void register_functions(Node *root) {
    for (int i = 0; i < root->count; i++) {
        Node *n = root->items[i];
        if (n->type != N_FN) continue;
        if (find_func(n->text))
            fatal_at(n->tok, "函数 '%s' 重复定义", n->text);
        if (nfuncs >= MAX_FUNCS)
            fatal_at(n->tok, "函数数量超出上限");
        snprintf(funcs[nfuncs].name, sizeof(funcs[nfuncs].name), "%s", n->text);
        funcs[nfuncs].decl = n;
        nfuncs++;
    }
}
```

**关键在于调用时机**：

```c
Node *prog = parse_program();

/* 先登记所有函数，再执行语句，这样函数之间可以互相调用 */
register_functions(prog);

Env global;
env_init(&global, NULL);
for (int i = 0; i < prog->count; i++) {
    Flow f = exec(prog->items[i], &global);
    ...
}
```

为什么必须**先全部登记，再执行**？看这个例子：

```
fn helper() { print(1); }
fn main() { helper(); }
main();
```

如果边执行边登记，执行到 `fn main()` 那条语句时，`helper` 还没被登记，
`main` 里调用 `helper` 就会报"未知函数 'helper'"。

**先登记再执行**还带来一个更重要的好处：**它是递归的前提**。
第 11 步的 `factorial` 会调用自己，而调用 `factorial` 时，
整个函数表已经填好了。

**已知限制**：`register_functions` 只扫描**顶层**语句。
写在块里的 `fn` 不会被登记：

```
{
    fn hidden() { print(1); }
}
hidden();          →  错误：未知函数 'hidden'
```

这是教学版的简化，真实语言里函数声明的作用域规则要复杂得多。

### 5.4 概念四：调用时开新作用域

```c
/* 每次调用都开一个新的作用域，参数就是里面的变量 */
Env local;
env_init(&local, e);
for (int i = 0; i < d->nparams; i++)
    env_push(&local, d->params[i], args[i]);

call_depth++;
Flow flow = exec(d->a, &local);
call_depth--;
env_free(&local);
```

**逐行解释**：

- `env_init(&local, e)` —— 新作用域的父环境是**调用者的作用域** `e`。
  所以函数体里能读到外层的变量（这叫**闭包**性质，MiniLang 提供了这个能力，
  虽然没有真正的"捕获"语义）。
- 参数被 `env_push` 进**参数作用域**。于是参数名在函数体里就是普通变量。
- `exec(d->a, &local)` —— 执行函数体（一个代码块）。
  函数体自己又会 `env_init` 出一个更内层的作用域（第 5 步的代码块逻辑）。
- `env_free(&local)` —— 调用结束释放参数作用域。
  和第 5 步一样，**每条返回路径都要释放**（本步只有一条）。

**参数遮蔽外层变量**：

```
let x = 1;
fn f(x) { print(x); }    ← 参数 x 遮蔽了全局的 x
f(99);                   →  99
```

因为查找变量是从内往外找的，参数作用域在最内层。

### 5.5 概念五：函数里 `let` 和 `=` 的区别

第 5 步讲过 `define_var`（只看当前作用域）和 `set_var`（沿链往外找）的区别。
在函数里，这个区别非常明显：

```
let g = 1;
fn bump() {
    g = g + 1;     /* 赋值：沿链找到全局的 g，改它 */
}
bump();
print(g);           /* → 2 */
```

```
let g = 1;
fn shadow() {
    let g = 2;     /* 定义：在函数的作用域里新建一个 g */
}
shadow();
print(g);          /* → 1，全局的 g 没被动过 */
```

**这就是 `case_06_mutate_outer.ml` 测的东西**：函数可以修改外层的变量。

### 5.6 概念六：`call_depth` 与无限递归

本步虽然还没有 `return`，但函数**已经可以调用自己**了：

```
fn loop() { loop(); }     /* 无限递归 */
loop();
```

没有保护的话，这就是无限套栈，最终段错误。所以要数一下当前调用了多少层：

```c
static int  call_depth;   /* 当前调用了多少层，用来防止无限递归 */

...

if (call_depth >= MAX_CALL_DEPTH)
    fatal_at(n->tok, "函数递归深度超过 %d", MAX_CALL_DEPTH);

call_depth++;
Flow flow = exec(d->a, &local);
call_depth--;
```

**代码要点**：

- `call_depth++` 和 `call_depth--` 必须**配对**，而且 `call_depth--`
  要在 `exec` 返回之后立刻执行（本步没有 `return` 提前退出的路径，所以没问题；
  第 11 步要检查这一点）。
- 检查放在**调用之前**，所以 `call_depth` 达到上限时报错。
- 这里的 `MAX_CALL_DEPTH = 128` 能不能真的达到，取决于第 5 步那个
  `Env` 动态数组的决定——**第 11 步会验证**。

### 5.7 概念七：C 语言的前向声明

本步的 `eval` 里要调用 `exec`，但 `exec` 定义在 `eval` **后面**：

```c
static Value eval(Node *n, Env *e) {
    ...
    Flow flow = exec(d->a, &local);   /* ← 这里要用 exec */
    ...
}

...

static Flow exec(Node *n, Env *e) {   /* ← 定义在后面 */
    ...
}
```

C 语言是"**先声明后使用**"的，所以要写一个**函数原型**（也叫前向声明）：

```c
/* exec 定义在 eval 后面，但 eval 里要调用它：
   C 语言要求"先声明后使用"，所以这里先写一个函数原型（也叫前向声明）。 */
static Flow exec(Node *n, Env *e);
```

**如果你写错了会怎样**：不写这个原型，C99 之前的编译器会报
"implicit declaration of function 'exec'"，C99 之后只是警告
（GCC/Clang 默认会警告），但**返回类型会默认为 `int`**——
而 `Flow` 是一个结构体！这么一搞，编译器会把"结构体被当整数用"当成合法代码，
最后得到一堆莫名其妙的错误。

第 13 步拆成多文件之后，这个原型会移到 `minilang.h` 里，
那时候它就不只是"前向声明"，而是正式的"模块间接口声明"。

## 6. 动手：把代码加进来

### 6.1 改动清单

| 位置 | 改了什么 | 行数 |
|---|---|---|
| 宏定义区 | 加 `MAX_FUNCS`、`MAX_CALL_DEPTH` | +2 |
| 错误处理区 | 加 `dupstr()` | +9 |
| `NodeType` 枚举 | 加 `N_FN` | +2 |
| `statement()` | 加 `fn` 分支 | +19 |
| `dump_ast()` | 加 `N_FN` 的 `case` | +7 |
| 新增"用户自定义函数"区 | `Func` 结构、`funcs[]`、`find_func()`、`register_functions()` | +34 |
| `eval()` | 加前向声明 | +4 |
| `eval()` 的 `N_CALL` | **改**：内置函数改成 `if (b) {...}`，后面接用户函数分支 | +32 / −14 |
| `exec()` | 加 `N_FN` 的 `case`（运行时什么都不用做） | +4 |
| `mode_run()` | 加 `register_functions(prog)` | +3 |

合计约 **+120 行**（其中有 14 行是"改"，见下面 6.2.4 的说明）。

### 6.2 分段讲解

#### 6.2.1 `dupstr` 和两个新宏

```c
#define MAX_FUNCS  256
#define MAX_CALL_DEPTH 128
```

```c
/* 复制一份字符串（参数名要保存到运行期，所以必须复制，不能指着源码） */
static char *dupstr(const char *s) {
    size_t n = strlen(s) + 1;
    char *p = (char *)malloc(n);
    if (!p) { perror("malloc"); exit(1); }
    memcpy(p, s, n);
    return p;
}
```

**逐行解释**：

- `size_t n = strlen(s) + 1;` —— `+1` 是给结尾的 `'\0'` 留位置。
  忘了 `+1` 的话，`memcpy` 会少拷一个字节，字符串没有终止符，
  后面用 `strcmp` 比较时会读到内存外面去。
- `if (!p) { perror("malloc"); exit(1); }` —— 分配失败要处理。
  `malloc` 返回 `NULL` 而不检查就直接 `memcpy` 进去，是最经典的崩溃写法。
- `memcpy(p, s, n)` 而不是 `strcpy(p, s)`：两者在这里等价，
  但 `memcpy` 把长度写明了，读代码时更清楚。

**放置位置**：`dupstr` 放在"错误处理"区之后、"标识符判断"之前——
它是个通用小工具，跟错误处理没关系，但放在文件靠前的位置，
后面用到的时候一眼就能找到。

#### 6.2.2 `fn` 的语法分析

```c
/* 函数定义：fn 名字(参数, ...) { ... } */
if (match(T_FN)) {
    Node *n = node(N_FN, t);
    Token id = expect(T_ID, "fn 后需要函数名");
    snprintf(n->text, sizeof(n->text), "%s", id.text);
    expect(T_LP, "函数参数列表需要 '('");
    if (cur()->type != T_RP) {
        do {
            if (n->nparams >= MAX_PARAMS)
                fatal_at(*cur(), "函数最多支持 %d 个参数", MAX_PARAMS);
            Token p = expect(T_ID, "参数必须是变量名");
            n->params[n->nparams++] = dupstr(p.text);
        } while (match(T_COMMA));
    }
    expect(T_RP, "参数列表需要 ')'");
    n->a = block();      /* 函数体也是一个代码块 */
    return n;
}
```

**逐行解释**：

- `expect(T_ID, "fn 后需要函数名")` —— 函数名必须是标识符。
  `fn 123() {}` 会报"语法错误：fn 后需要函数名，当前是 '123'"。
- `if (cur()->type != T_RP)` —— 零参数函数。
- `do { ... } while (match(T_COMMA));` —— 至少处理一个参数，
  之后每有一个逗号就再处理一个。
  `match()` 顺便把逗号吃掉了，所以循环体里不用管逗号。
- `n->a = block()` —— **函数体复用代码块解析器**。
  这是本步最省事的地方：函数体的作用域、语句列表，全部免费获得。

**如果你写错了会怎样**：忘了 `n->a = block()`，函数体就不会被解析，
调用时 `exec(NULL, ...)` 直接返回，函数变成空函数——而且**不报任何错**，
非常难查。

#### 6.2.3 函数表和登记

```c
typedef struct {
    char  name[MAX_NAME];  /* 函数名     */
    Node *decl;            /* 函数声明节点（名字、参数、函数体都在里面） */
} Func;

static Func funcs[MAX_FUNCS];
static int  nfuncs;
static int  call_depth;

static Func *find_func(const char *name) {
    for (int i = 0; i < nfuncs; i++)
        if (!strcmp(funcs[i].name, name)) return &funcs[i];
    return NULL;
}
```

**逐行解释**：

- `find_func` 返回 `Func *`（指向表里的元素）而不是拷贝一份。
  这样调用时能拿到 `decl` 指针。
- **注意这个 `find_func` 是正着找的**，和变量查找（`find_var` 倒着找）不一样。
  因为函数不允许重复定义（登记时会检查），正着找反着找一样。

```c
static void register_functions(Node *root) {
    for (int i = 0; i < root->count; i++) {
        Node *n = root->items[i];
        if (n->type != N_FN) continue;
        if (find_func(n->text))
            fatal_at(n->tok, "函数 '%s' 重复定义", n->text);
        if (nfuncs >= MAX_FUNCS)
            fatal_at(n->tok, "函数数量超出上限");
        snprintf(funcs[nfuncs].name, sizeof(funcs[nfuncs].name), "%s", n->text);
        funcs[nfuncs].decl = n;
        nfuncs++;
    }
}
```

**逐行解释**：

- `if (n->type != N_FN) continue;` —— 只处理函数定义，其它语句跳过。
- 重复定义检查放在**登记之前**，所以报的是"重复定义"而不是后一次覆盖前一次。
- `nfuncs++` —— 登记完一个就加一。

#### 6.2.4 `N_CALL` 分支的改造（本步唯一"改既有代码"的地方）

第 9 步的 `N_CALL` 长这样：

```c
const Builtin *b = find_builtin(n->text);
if (!b)
    fatal_at(n->tok, "未知函数 '%s'", n->text);
if (n->count < b->min_args || n->count > b->max_args)
    fatal_at(...);
double args[MAX_PARAMS];
for (...) { ... }
return number(b->fn(args, n->tok));
```

"查不到内置函数就报错"——现在得多一条路：**查不到内置函数，就查用户函数**。
所以要把结构改成：

```diff
--- a/minilang.c
+++ b/minilang.c
@@ -33,6 +33,8 @@
 #define MAX_PARAMS 32
 #define MAX_VARS   2048
 #define MAX_NAME   128
+#define MAX_FUNCS  256
+#define MAX_CALL_DEPTH 128
 #define MAX_LOOP_STEPS 1000000
 
 /* ------------------------------------------------------------------ */
@@ -81,6 +83,15 @@
     fputc('\n', stderr);
     had_error = 1;
     exit(1);
+}
+
+/* 复制一份字符串（参数名要保存到运行期，所以必须复制，不能指着源码） */
+static char *dupstr(const char *s) {
+    size_t n = strlen(s) + 1;
+    char *p = (char *)malloc(n);
+    if (!p) { perror("malloc"); exit(1); }
+    memcpy(p, s, n);
+    return p;
 }
 
 /* 标识符可以以字母或下划线开头 */
@@ -275,7 +286,8 @@
     N_ASSIGN,   /* x = 1;            */
     N_EXPR,     /* 表达式语句        */
     N_IF,       /* if (条件) ... else ... */
-    N_FOR       /* for (init; cond; update) { ... } */
+    N_FOR,      /* for (init; cond; update) { ... } */
+    N_FN        /* fn 名字(参数) { ... } */
 } NodeType;
 
 typedef struct Node Node;
@@ -456,6 +468,25 @@
     /* 裸代码块 { ... } */
     if (cur()->type == T_LB) return block();
 
+    /* 函数定义：fn 名字(参数, ...) { ... } */
+    if (match(T_FN)) {
+        Node *n = node(N_FN, t);
+        Token id = expect(T_ID, "fn 后需要函数名");
+        snprintf(n->text, sizeof(n->text), "%s", id.text);
+        expect(T_LP, "函数参数列表需要 '('");
+        if (cur()->type != T_RP) {
+            do {
+                if (n->nparams >= MAX_PARAMS)
+                    fatal_at(*cur(), "函数最多支持 %d 个参数", MAX_PARAMS);
+                Token p = expect(T_ID, "参数必须是变量名");
+                n->params[n->nparams++] = dupstr(p.text);   /* 参数名要留到运行时用，复制一份 */
+            } while (match(T_COMMA));
+        }
+        expect(T_RP, "参数列表需要 ')'");
+        n->a = block();      /* 函数体也是一个代码块 */
+        return n;
+    }
+
     /* if (条件) 语句 [else 语句] */
     if (match(T_IF)) {
         Node *n = node(N_IF, t);
@@ -617,6 +648,13 @@
             if (n->c) kids[m++] = n->c;   /* 更新 */
             for (int i = 0; i < n->count; i++) kids[m++] = n->items[i];  /* 循环体 */
             break;
+        case N_FN:
+            printf("函数 %s(", n->text);
+            for (int i = 0; i < n->nparams; i++)
+                printf("%s%s", i ? ", " : "", n->params[i]);
+            printf(")\n");
+            kids[m++] = n->a;   /* 函数体 */
+            break;
     }
 
     for (int i = 0; i < m; i++) {
@@ -672,6 +710,40 @@
 }
 
 /* ------------------------------------------------------------------ */
+/* 用户自定义函数                                                      */
+/* ------------------------------------------------------------------ */
+typedef struct {
+    char  name[MAX_NAME];  /* 函数名     */
+    Node *decl;            /* 函数声明节点（名字、参数、函数体都在里面） */
+} Func;
+
+static Func funcs[MAX_FUNCS];
+static int  nfuncs;
+static int  call_depth;   /* 当前调用了多少层，用来防止无限递归 */
+
+static Func *find_func(const char *name) {
+    for (int i = 0; i < nfuncs; i++)
+        if (!strcmp(funcs[i].name, name)) return &funcs[i];
+    return NULL;
+}
+
+/* 登记程序里定义的所有函数。
+   必须在执行任何语句之前做完，这样函数之间可以互相调用、也可以递归。 */
+static void register_functions(Node *root) {
+    for (int i = 0; i < root->count; i++) {
+        Node *n = root->items[i];
+        if (n->type != N_FN) continue;
+        if (find_func(n->text))
+            fatal_at(n->tok, "函数 '%s' 重复定义", n->text);
+        if (nfuncs >= MAX_FUNCS)
+            fatal_at(n->tok, "函数数量超出上限");
+        snprintf(funcs[nfuncs].name, sizeof(funcs[nfuncs].name), "%s", n->text);
+        funcs[nfuncs].decl = n;
+        nfuncs++;
+    }
+}
+
+/* ------------------------------------------------------------------ */
 /* 运行时：值与求值                                                    */
 /* ------------------------------------------------------------------ */
 
@@ -813,6 +885,10 @@
     return f;
 }
 
+/* exec 定义在 eval 后面，但 eval 里要调用它：
+   C 语言要求"先声明后使用"，所以这里先写一个函数原型（也叫前向声明）。 */
+static Flow exec(Node *n, Env *e);
+
 /* 求值：把一棵表达式树算成一个值。
    递归的结构和语法分析器正好相反：分析时"从左到右展开"，
    求值时"从叶子往回算"。 */
@@ -853,20 +929,52 @@
 
             /* 其余内建函数：查表 */
             const Builtin *b = find_builtin(n->text);
-            if (!b)
+            if (b) {
+                if (n->count < b->min_args || n->count > b->max_args)
+                    fatal_at(n->tok, "函数 '%s' 参数数量错误：需要 %d 个，实际 %d 个",
+                             b->name, b->min_args, n->count);
+                double args[MAX_PARAMS];
+                for (int i = 0; i < n->count; i++) {
+                    Value v = eval(n->items[i], e);
+                    if (v.is_string)
+                        fatal_at(n->tok, "函数 '%s' 的参数必须是数字", b->name);
+                    args[i] = v.value;
+                }
+                return number(b->fn(args, n->tok));
+            }
+
+            /* 用户定义的函数 */
+            Func *f = find_func(n->text);
+            if (!f)
                 fatal_at(n->tok, "未知函数 '%s'", n->text);
-            if (n->count < b->min_args || n->count > b->max_args)
-                fatal_at(n->tok, "函数 '%s' 参数数量错误：需要 %d 个，实际 %d 个",
-                         b->name, b->min_args, n->count);
+            Node *d = f->decl;
+            if (n->count != d->nparams)
+                fatal_at(n->tok, "函数 '%s' 需要 %d 个参数，实际传入 %d 个",
+                         n->text, d->nparams, n->count);
+            if (call_depth >= MAX_CALL_DEPTH)
+                fatal_at(n->tok, "函数递归深度超过 %d", MAX_CALL_DEPTH);
 
             double args[MAX_PARAMS];
             for (int i = 0; i < n->count; i++) {
                 Value v = eval(n->items[i], e);
                 if (v.is_string)
-                    fatal_at(n->tok, "函数 '%s' 的参数必须是数字", b->name);
+                    fatal_at(n->tok, "函数参数必须是数字");
                 args[i] = v.value;
             }
-            return number(b->fn(args, n->tok));
+
+            /* 每次调用都开一个新的作用域，参数就是里面的变量 */
+            Env local;
+            env_init(&local, e);
+            for (int i = 0; i < d->nparams; i++)
+                env_push(&local, d->params[i], args[i]);
+
+            call_depth++;
+            Flow flow = exec(d->a, &local);
+            call_depth--;
+            env_free(&local);
+
+            /* 还没有 return，所以函数的值暂时是 0（下一步会补上） */
+            return flow.returned ? flow.value : number(0);
         }
 
         case N_UNARY: {
@@ -968,6 +1076,10 @@
             /* 没有 else 时 n->c 是 NULL，exec(NULL) 直接返回"什么都没发生" */
         }
 
+        case N_FN:
+            /* 函数定义在执行前就登记好了，这里什么都不用做 */
+            return f;
+
         case N_FOR: {
             /* 循环自带一个作用域：i 只在循环里活着 */
             Env local;
@@ -1067,6 +1179,9 @@
     lex();
     Node *prog = parse_program();
 
+    /* 先登记所有函数，再执行语句，这样函数之间可以互相调用 */
+    register_functions(prog);
+
     Env global;
     env_init(&global, NULL);
     for (int i = 0; i < prog->count; i++) {
```

**逐行解释**：

- `if (b) { ... return ...; }` —— 内置函数处理完就 `return`，
  所以下面的用户函数分支只有在"不是内置函数"时才会执行。
- 结尾的 `fatal_at(n->tok, "未知函数 '%s'", n->text)` **保留在最后**，
  意思是"既不是内置函数，也不是用户函数"——这才是真正的"未知函数"。

**这是本教程的第 3 处"改既有代码"**（前两处是第 9 步的字符串模板）。
改动是必要的：`N_CALL` 从"一条路"变成"两条路"，控制流必须调整。
好在改的是**一个 `if` 的形状**，函数体里的逻辑一行没动。

#### 6.2.5 调用用户函数

```c
/* 用户定义的函数 */
Func *f = find_func(n->text);
if (!f)
    fatal_at(n->tok, "未知函数 '%s'", n->text);
Node *d = f->decl;
if (n->count != d->nparams)
    fatal_at(n->tok, "函数 '%s' 需要 %d 个参数，实际传入 %d 个",
             n->text, d->nparams, n->count);
if (call_depth >= MAX_CALL_DEPTH)
    fatal_at(n->tok, "函数递归深度超过 %d", MAX_CALL_DEPTH);

double args[MAX_PARAMS];
for (int i = 0; i < n->count; i++) {
    Value v = eval(n->items[i], e);
    if (v.is_string)
        fatal_at(n->tok, "函数参数必须是数字");
    args[i] = v.value;
}

/* 每次调用都开一个新的作用域，参数就是里面的变量 */
Env local;
env_init(&local, e);
for (int i = 0; i < d->nparams; i++)
    env_push(&local, d->params[i], args[i]);

call_depth++;
Flow flow = exec(d->a, &local);
call_depth--;
env_free(&local);

/* 还没有 return，所以函数的值暂时是 0（下一步会补上） */
return flow.returned ? flow.value : number(0);
```

**逐行解释**：

- `Node *d = f->decl;` —— 取出声明节点。参数个数在 `d->nparams`，
  参数名在 `d->params[]`，函数体在 `d->a`。
- **参数个数检查在求值实参之前**：这样 `add2(1)` 会报"需要 2 个参数，实际传入 1 个"，
  而不是先算出 `1` 再报错。顺序不同，报错信息不同。
- 实参**在调用者的作用域里求值**（`eval(n->items[i], e)`，注意是 `e` 不是 `local`），
  这是正确的：实参表达式应该看到调用者的变量。
- `env_init(&local, e)` —— 新作用域的父是调用者。
- `call_depth++` / `call_depth--` 夹住 `exec`。
- `return flow.returned ? flow.value : number(0);` —— **本步的"半成品"**：
  `flow.returned` 现在永远是 0（第 11 步才有 `return`），
  所以函数的值永远是 0。这个三元表达式先写好，第 11 步就只需要改 `exec` 里的 `N_RETURN`。

**如果你写错了会怎样**：把 `eval(n->items[i], e)` 写成 `eval(n->items[i], &local)`，
实参就只能在参数作用域里求值——而那时参数还没填进去，
`fn f(x) { return g(x); }` 这种"用参数当实参"的情况就会找不到变量。

#### 6.2.6 `exec` 里的 `N_FN`

```c
case N_FN:
    /* 函数定义在执行前就登记好了，这里什么都不用做 */
    return f;
```

**逐行解释**：函数定义是一条**语句**，出现在程序的顶层。
但它真正"生效"是在 `register_functions` 那一步。执行到这里时什么都不用做，
直接返回"什么都没发生"。

**如果你写错了会怎样**：忘了这个 `case`，`exec` 会走到 `default` 分支报
"这里不是一条语句"——虽然语句本身是合法的。

### 6.3 完整 diff

> **怎么用这段 diff**：推荐**照着敲**——每改一小段就编译一次，比一次性粘贴更容易记住。
> 如果想直接应用，把下面的内容存成 `step.patch`，然后在**上一步的目录**里执行：
>
> ```sh
> patch -p1 < ../010/step.patch
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
| `MAX_FUNCS` / `MAX_CALL_DEPTH` | `#define` 区，`MAX_NAME` 之后 |
| `dupstr()` | "错误处理"区之后，"标识符可以以字母开头"之前 |
| `N_FN` | `NodeType` 枚举末尾 |
| `fn` 分支 | `statement()` 里，裸代码块分支之后 |
| `N_FN` 的 `case` | `dump_ast()` 的 `switch` 末尾 |
| `Func` / `funcs[]` / `find_func` / `register_functions` | **新增一整节**，放在"运行时：值与求值"之前 |
| `static Flow exec(Node *n, Env *e);` | `eval()` 之前 |
| `N_CALL` 的改造 | `eval()` 的 `case N_CALL:` 里 |
| `case N_FN:` | `exec()` 的 `switch` 里，`case N_IF` 之后 |
| `register_functions(prog);` | `mode_run()` 里，`parse_program()` 之后 |

## 7. 测试

`./run.sh` 的真实输出：

```
▶ 用例（模式：run）
  ✅ case_01_function_call.ml     [run]
  ✅ case_02_with_args.ml         [run]
  ✅ case_03_wrong_arity.ml       [run]
  ✅ case_04_unknown_function.ml  [run]
  ✅ case_05_duplicate.ml         [run]
  ✅ case_06_mutate_outer.ml      [run]
▶ demo.ml
  | 7
✅ 全部通过（7 项）
```

逐个用例说明：

### case_01：无参函数

```
fn show() { print(1); }
show();
```

输出 `1`。

### case_02：有参函数

```
fn show2(a, b) { print(a + b); }
show2(3, 4);
```

输出 `7`。实参 `3`、`4` 在**调用者的作用域**里求值（都是数字，直接就是 7），
然后被 `env_push` 进参数作用域。

### case_03：参数个数不对（错误用例）

```
fn add2(a, b) { print(a + b); }
add2(1);
```

期望 stderr：

```
错误 [2:1]: 函数 'add2' 需要 2 个参数，实际传入 1 个
```

期望退出码 `1`。位置是**调用点**（第 2 行第 1 列），不是函数定义处。

### case_04：未知函数（错误用例）

```
nosuch(1);
```

期望 stderr：

```
错误 [1:1]: 未知函数 'nosuch'
```

期望退出码 `1`。注意这条消息和第 8 步的**完全一样**——
因为"既不是内置函数也不是用户函数"就是未知函数。

### case_05：重复定义（错误用例）

```
fn f() { print(1); }
fn f() { print(2); }
```

期望 stderr：

```
错误 [2:1]: 函数 'f' 重复定义
```

期望退出码 `1`。位置是**第二个** `fn` 的位置。

### case_06：函数修改外层变量

```
let g = 1;
fn bump() { g = g + 1; }
bump();
print(g);
```

输出 `2`。函数里的 `g = g + 1` 是**赋值**（`set_var`），
沿作用域链找到全局的 `g` 并修改它。

### 自己动手试

```sh
cd 010
./minilang demo.ml ast
```

```
语法树：
代码块
├── 函数 add(a, b)
│   └── 代码块
│       └── 表达式语句
│           └── 调用 print
│               └── 二元 PLUS
│                   ├── 变量 a
│                   └── 变量 b
└── 表达式语句
    └── 调用 add
        ├── 数字 3
        └── 数字 4
```

## 8. 易错点与坑

### 坑 1：忘记写前向声明

**现象**：编译报 "implicit declaration of function 'exec'"，
或者更糟——**编译过了但运行是乱的**。

**原因**：C99 之后 `exec` 没被声明时，编译器默认它返回 `int`。
而 `exec` 实际返回 `Flow`（结构体），调用方会按 `int` 去解读返回的内存，
得到完全无法预测的行为。

**正确写法**：写 `static Flow exec(Node *n, Env *e);`。
第 13 步它会变成头文件里的正式声明。

### 坑 2：把 `dupstr` 写成 `strdup`

**现象**：macOS 上能编译，Linux 上编译报错。

**原因**：`strdup` 是 POSIX 标准，不是 C 标准。Linux 的 glibc 需要
`_POSIX_C_SOURCE` 或 `_DEFAULT_SOURCE` 才暴露它，而 macOS 默认就有。

**正确写法**：自己写一个 `dupstr`（本步的做法），或者在 Linux 上加
`-D_GNU_SOURCE`。自己写只有 6 行，还顺便处理了分配失败。

### 坑 3：参数求值用错了作用域

**现象**：`fn f(x) { return g(x); }` 报"未定义变量 'x'"。

**原因**：实参求值用了参数作用域（那时参数还没填）。

**正确写法**：实参一律用**调用者的作用域**求值：`eval(n->items[i], e)`。

### 坑 4：`call_depth++` 和 `call_depth--` 没配对

**现象**：递归调用几次之后，莫名其妙地报"递归深度超过 128"。

**原因**：某条路径上 `call_depth--` 被跳过了。

**正确写法**：`call_depth++` 之后紧跟的必须是对应的 `--`。
第 11 步加 `return` 之后要**重新检查一遍**所有提前返回的路径。

### 坑 5：函数定义写在块里

**现象**：块里定义的函数调用时报"未知函数"。

**原因**：`register_functions` 只扫描顶层语句。

**正确写法**：教学版就把函数定义写在顶层。真要支持块内定义，
得把 `register_functions` 改成递归扫描，并且想清楚嵌套函数的作用域规则。

### 坑 6：`Func` 表满了以后越界写

**现象**：定义 300 个函数后崩溃或行为异常。

**原因**：`funcs[MAX_FUNCS]` 是固定数组。

**正确写法**：登记前检查 `if (nfuncs >= MAX_FUNCS)`——本步已经加了，
所以超限时是**友好报错**"函数数量超出上限"，而不是段错误。

## 9. 自测题

### 9.1 概念题

**Q1.** 为什么函数表要在执行语句**之前**填好？

<details><summary>参考答案</summary>

因为函数之间可以互相调用（包括递归）。如果边执行边登记，
先被调用的函数可能还没被登记，报"未知函数"。
先登记再执行保证了"执行任何语句时，所有函数都已可见"。

</details>

**Q2.** `Func` 结构体里为什么只存名字和声明节点的指针，不把参数列表也拷一份？

<details><summary>参考答案</summary>

参数名和函数体都在 `N_FN` 节点里（`params[]` 和 `a`），
再拷一份就是重复存储，而且两份数据可能不一致。
只存指针，天然同步。

</details>

**Q3.** 函数里的 `g = g + 1` 和 `let g = 1` 对外层的 `g` 有什么不同？

<details><summary>参考答案</summary>

`g = g + 1` 是赋值（`set_var`），沿作用域链找到外层的 `g` 并修改它；
`let g = 1` 是定义（`define_var`），在函数的作用域里新建一个 `g`，外层不受影响。

</details>

### 9.2 预测题（改一行会发生什么）

**Q4.** 把 `register_functions(prog)` 挪到执行语句的循环**之后**，`case_03` 会怎样？

<details><summary>参考答案</summary>

`case_03` 是错误用例（参数个数不对），还是报错，所以看不出差别。
但 `demo.ml` 会挂：执行 `add(3, 4)` 时函数表还是空的，
报"未知函数 'add'"。

</details>

**Q5.** 把 `env_init(&local, e)` 改成 `env_init(&local, NULL)`，函数体会怎样？

<details><summary>参考答案</summary>

函数体里**看不到任何外层变量**了（作用域链断开了）。
`fn add(a, b) { print(a + b); }` 还能用（参数在自己的作用域里），
但函数体里引用全局变量就会报"未定义变量"。

</details>

**Q6.** 把 `if (n->count != d->nparams)` 这个检查删掉，会发生什么？

<details><summary>参考答案</summary>

`add2(1)` 调用时只 push 了 1 个参数，但函数体里引用 `b` 时
会在参数作用域里找不到 `b`——**沿链往外找也找不到**（外层没有 `b`），
于是报"未定义变量 'b'"。错误信息不如现在这条准确，但至少不会崩。

</details>

### 9.3 动手题

**Q7.** 写一个 `fn max2(a, b)`，用 `if` 实现（`if` 第 6 步就有了）：

```
fn max2(a, b) {
    if (a > b) { print(a); } else { print(b); }
}
max2(3, 9);
```

预期输出 `9`。

<details><summary>参考答案</summary>

`max2(3, 9)` 输出 `9`。
注意函数体里的 `if` 是第 6 步的语法，`a` 和 `b` 是参数作用域里的变量。

</details>

**Q8.** 写一个函数，把外层的一个变量改成传入的值：

```
let counter = 0;
fn set(v) { counter = v; }
set(42);
print(counter);
```

预期输出 `42`。

<details><summary>参考答案</summary>

输出 `42`。`counter = v` 是赋值，沿链找到全局的 `counter` 并修改。
这和 `case_06` 是同一个机制。

</details>

**Q9.** 把 `MAX_CALL_DEPTH` 从 128 改成 3，然后跑 `fn loop() { loop(); }`，
看看报错和退出码。

<details><summary>参考答案</summary>

```
错误 [1:14]: 函数递归深度超过 3
```

退出码 `1`。
注意本步还没有 `return`，所以 `loop()` 永远执行不完，
是 `call_depth` 计数器**主动**终止了它——这正是它的作用。

</details>

### 9.4 延伸题

**Q10.** 如果想让函数可以返回"多个值"（比如同时返回 a 和 b），你打算怎么改？

<details><summary>参考答案</summary>

`Value` 结构体里已经有 `is_string` 和 `string[MAX_TEXT]`，
可以在此基础上扩展成一个"联合体"（union）风格的返回值：
比如 `Value` 增加一个 `double pair[2]` 字段，用 `is_pair` 标记。
或者更简单：让函数返回一个数组（需要先给语言加数组类型）。

</details>

## 10. 本步的设计取舍

- **函数表用固定数组**：和 `Token`/`Node` 一样的风格，简单直接，代价是上限 256 个函数。
- **`Func` 只存指针**：避免重复存储，但意味着函数表里的东西都指向 AST 节点，
  AST 节点不能被提前释放（本步是静态池，正好）。
- **本步不给函数返回值**：看起来"半成品"，但这其实是**为下一步做准备**——
  第 11 步只要改 `exec` 的一个 `case`，`eval` 里这行
  `return flow.returned ? flow.value : number(0);` 一个字都不用动。
- **`dupstr` 的内存不回收**：教学版的已知泄漏。正式实现应该用 arena。

## 11. 下一步预告

函数能定义了、能调用了，但**没法把结果算出来传回去**——
`fn add(a, b) { print(a + b); }` 只能打印，不能"返回"。

下一节：[第 11 步 · return 与递归](../011/README.md)，我们会加上：

- `return 表达式;` 和 `return;` 两种形式；
- 第 5 步定型的 `Flow` 结构**正式上线**——`exec()` 的签名一个字都不用改；
- 递归：`factorial(5)`、`fib(10)`；
- **第 5 步那个动态数组决定的回报**——递归 100 层不崩，
  而固定数组版本第 8 层就段错误（`011/experiments/` 里有现成的对照实验）。
