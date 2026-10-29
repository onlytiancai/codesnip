# 第 7 步 · for 循环

> 进度 ███████░░░░░ 7/13 ｜ 上一节：[第 6 步 · if / else 条件分支](../006/README.md) ｜ 下一节：[第 8 步 · 内置函数](../008/README.md)

## 1. 本步目标

读完你将能够：

- 让程序重复执行一段代码：`for (let i = 1; i <= 10; i = i + 1) { total = total + i; }` 能算出 1~10 的和。
- 说清楚 C 式 `for` 的三个部分（初始化 / 条件 / 更新）分别对应 AST 里的哪几个字段。
- 把 `for` 展开成等价的 `while`，并解释 `for` 语句本身**为什么不需要分号结尾**。
- 用一个"忘记写更新部分"的 `for` 复现死循环，亲眼看到安全阀在 100 万步后报错退出。
- 解释为什么循环变量 `i` 在循环外面就"不存在了"。

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
| `tokens` | 任意 `.ml` | token 列表（`FOR` 关键字从第 1 步就在词法表里） | 0（有错为 1） |
| `ast` | 任意 `.ml` | 语法树，`for` 打印成 `循环` 节点，4 个子节点 | 0（有错为 1） |
| `run`（默认） | 任意 `.ml` | 真正执行循环；死循环会在 100 万步后报错退出 | 0；错误为 1 |

`007/demo.ml` 的内容：

```
// 第 7 步：循环
let total = 0;
for (let i = 1; i <= 10; i = i + 1) {
    total = total + i;
}
print(total);
```

真实运行结果：

```sh
$ ./minilang demo.ml
55
```

用 `ast` 模式看这棵树：

```sh
$ ./minilang demo.ml ast
语法树：
代码块
├── 定义 total
│   └── 数字 0
├── 循环
│   ├── 定义 i
│   │   └── 数字 1
│   ├── 二元 LE
│   │   ├── 变量 i
│   │   └── 数字 10
│   ├── 赋值 i
│   │   └── 二元 PLUS
│   │       ├── 变量 i
│   │       └── 数字 1
│   └── 代码块
│       └── 赋值 total
│           └── 二元 PLUS
│               ├── 变量 total
│               └── 变量 i
└── 表达式语句
    └── 调用 print
        └── 变量 total
```

`循环` 节点下面正好 4 个子节点：**初始化、条件、更新、循环体**。这四个正好对应 AST 里的 `a`、`b`、`c` 三个指针加上 `items[0]`。

## 4. 前置知识回顾

上一步我们已经学会了写 `if`。本步要用的东西有三样：

**（1）`N_IF` 节点的三段式。** 上一步我们把 `if (条件) then else` 拆成 `a`（条件）、`b`（then）、`c`（else）。`for` 的结构惊人地相似：**头部三段 + 一个循环体**。所以 AST 里同样是"几个通用子指针"，不需要动 `Node` 结构体（INV-5）。

**（2）`exec()` 的返回值是 `Flow`。** 第 5 步定型的 `Flow { int returned; Value value; }`，本步的循环体执行完后要检查 `f.returned`——虽然第 7 步还没有 `return` 语句（那是第 11 步），但 `Flow` 这个字段从第一天就留着，正好用上。

**（3）`Env` 作用域链。** 第 5 步引入的 `env_init` / `env_free` / `env_push` / `find_var` / `get_var` / `set_var`。本步的循环要**自带一个作用域**，实现方式和代码块一模一样：`env_init(&local, e)` 建一个父作用域是 `e` 的新环境，循环结束后 `env_free(&local)` 释放。

## 5. 本步引入的核心概念

### 5.1 概念一：C 式 `for` 的三段分别对应 AST 的哪几个字段

**问题**：`for (初始化; 条件; 更新) 循环体` 有四个部分，AST 节点要放哪儿？

**思路**：`Node` 结构体已经有 `a`、`b`、`c` 三个子指针和一个 `items[]` 数组。分配如下：

```
              N_FOR
             /  |  |  \
            a   b  c  items[0]
            │   │  │     │
          初始化 条件 更新  循环体
```

| 源码部分 | AST 字段 | 节点类型 | 可以为空吗 |
|---|---|---|---|
| `for (` 之后、`;` 之前的初始化 | `n->a` | `N_LET` 或 `N_ASSIGN` | **可以**（`for (; ...)`） |
| 第一个 `;` 和第二个 `;` 之间的条件 | `n->b` | 任意表达式节点（`N_BINARY` / `N_VAR` / …） | **可以**（留空 = 永远为真） |
| 第二个 `;` 和 `)` 之间的更新 | `n->c` | `N_ASSIGN` | **可以**（`for (...;)`） |
| `)` 之后的循环体 | `n->items[0]` | 任意语句节点（通常是 `N_BLOCK`） | 不可以 |

**代码要点**：`n->items[n->count++] = statement();` —— 循环体被塞进 `items` 数组而不是 `a`/`b`/`c`。为什么不用 `d`？因为 `Node` 里没有第四个通用指针，而 `items[128]` 这个数组本来就是给"一串子节点"用的（第 2 步给 `N_CALL` 的实参列表、第 5 步给 `N_BLOCK` 的语句列表）。循环体虽然只有一个，但复用 `items` 更自然——`dump_ast` 里也只要写一个 `for` 循环就能把循环体打出来。

**边界情况**：`n->a` / `n->b` / `n->c` 都可能是 `NULL`，`dump_ast` 里必须逐个判空。

### 5.2 概念二：`for` 展开成等价的 `while`

**问题**：C 的 `for` 是个语法糖。展开之后是什么？

**思路**：展开成"一条初始化语句 + 一个 `while` + 一次更新"：

```
   for (init; cond; update) BODY
   ↓
   init;
   while (cond) {
       BODY;
       update;
   }
```

展开成 while 循环后，语义就一目了然了：

- `init` 在循环**之前**执行**一次**。
- `cond` 在每次迭代**开始前**检查一次；为假就退出。
- `BODY` 执行一次。
- `update` 在 BODY **之后**执行一次。
- 回到 `cond`。

用 `tests/case_01_sum.ml` 对照：

```
let total = 0;
for (let i = 1; i <= 5; i = i + 1) {
    total = total + i;
}
```

展开后是：

```
let total = 0;
let i = 1;                  ← init，只执行一次
while (i <= 5) {            ← cond，每轮开头检查
    total = total + i;      ← BODY
    i = i + 1;              ← update，每轮结尾
}
print(total);
```

**代码要点**：这个展开就是 `exec` 里 `case N_FOR:` 的代码结构——**`init` 在 `while` 之前，`update` 在 `while` 的循环体最后**。你写的 C 代码和这个展开一一对应。

**边界情况**：条件留空（`for (;;)`）等价于 `while (1)`，也就是**死循环**。MiniLang 靠安全阀兜底（见 5.5）。

### 5.3 概念三：`for` 语句本身不需要分号结尾

**问题**：`for (let i = 0; i < 3; i = i + 1) { print(i); }` 后面要不要加分号？

**思路**：**不要**。这是 MiniLang 里最容易搞混的一个"分号"：

```
   for ( 初始化 ; 条件 ; 更新 )  循环体
       ↑        ↑        ↑
       这三个分号是 for 头部的一部分，分隔头部和循环体
              for 语句本身结尾不需要分号
```

为什么？因为 `for` 的头部**已经用完了两个分号**（三个部分的分隔符），C 语言里 `for` 头部本来就不需要额外的分号——`for (init; cond; upd) stmt` 整体就是一条语句。MiniLang 保持一致。

**代码要点**：在解析器里，`for` 的三个部分各自"吃掉"自己的分号：

- 初始化部分末尾 `expect(T_SEMI, "for 初始化后需要 ';'")`；
- 条件部分末尾 `expect(T_SEMI, "for 条件后需要 ';'")`；
- 更新部分**没有**分号，它后面直接是 `)`：`expect(T_RP, "for 头部需要 ')'")`。

**边界情况**：写多了会怎样？

```sh
$ cat /tmp/trailing_semi.ml
for (let i = 0; i < 3; i = i + 1) { print(i); };
$ ./minilang /tmp/trailing_semi.ml
错误 [1:48]: 语法错误：此处需要数字、变量、函数调用或括号表达式
```

MiniLang **没有空语句**（C 里的 `;`）。所以多出来的那个 `;` 会走 `statement()` 的最后一条路（"其它都是表达式语句"），然后 `expression()` 看到 `;` 就报错了。

而如果**少写**头部里的分号，比如 `for (let i = 0 i < 3; ...)`，报错是：

```
错误 [行:列]: 语法错误：for 初始化后需要 ';'
```

### 5.4 概念四：`for` 自带作用域

**问题**：

```
for (let i = 0; i < 2; i = i + 1) { print(i); }
print(i);
```

第二行会打印什么？

**思路**：报"未定义变量 'i'"。因为 `for` 的执行会建一个**自己的** `Env`：

```c
case N_FOR: {
    Env local;
    env_init(&local, e);      /* local 的 parent 是 e */
    ...
    env_free(&local);         /* 循环结束就销毁 */
    return noflow();
}
```

初始化里的 `let i = 0` 把 `i` 定义在 `local` 里；`local` 是 `env_init(&local, e)` 建的，父作用域是外层 `e`。循环体是一个 `N_BLOCK`，块自己又会建一个 `Env`，父作用域是 `&local`——所以循环体里能看见 `i`。循环一结束，`env_free(&local)` 把 `i` 一起释放，循环外就再也看不见了。

```
        外层环境 e
        ┌─────────────────┐
        │                 │
        │  Env local      │   ← i 定义在这里（for 初始化里的 let）
        │  ┌───────────┐  │
        │  │ Env block │  │   ← 循环体是块，块自己的环境，parent = local
        │  │  能看见 i │  │
        │  └───────────┘  │
        └─────────────────┘
             循环结束 → env_free(&local) → i 消失
```

**代码要点**：注意 `local` 的 `parent` 是 `e`，**不是** `&local` 自己。`env_init` 传进去的第二个参数就是父指针。

**边界情况（非常重要）**：**赋值**和**定义**的查找范围不一样！

- `define_var`（`let`）只在**当前层**找，找不到就报错"变量 'x' 在当前作用域中已定义"——`let` 永远新建变量。
- `set_var`（`x = ...`）沿着 `parent` 链**往外找**，找到第一个同名变量就改它。

所以：

```
let i = 0;
for (; i < 3; i = i + 1) { print(i); }
print(i);
```

更新部分 `i = i + 1` 是**赋值**，它从 `local` 开始往外找，在外层找到了 `i`，于是改的是**外层的 `i`**。真实输出：

```
0123
```

（循环里打印 0、1、2，循环外 `i` 已经是 3。）

### 5.5 概念五：安全阀 `MAX_LOOP_STEPS`

**问题**：解释器跑 `while (1) { }` 会怎样？——永远不结束，你的终端被卡死，只能 `Ctrl-C`。

**思路**：加一个**全局步数计数器**，超过上限就报错退出：

```c
#define MAX_LOOP_STEPS 1000000
...
static int loop_steps;   /* 循环一共执行了多少步（用来防死循环） */
```

在循环体的开头检查：

```c
if (++loop_steps > MAX_LOOP_STEPS) {
    env_free(&local);
    fatal_at(n->tok, "循环执行超过安全上限 %d 步", MAX_LOOP_STEPS);
}
```

**代码要点**：

- `++loop_steps` 是**先加后比较**，所以第 1000001 次迭代触发。
- `env_free(&local)` 要在报错前调用——虽然 `fatal_at` 直接 `exit(1)`，内存会被 OS 回收，但这是好习惯（也让代码在将来改成"跳出不退出"时依然正确）。
- `fatal_at(n->tok, ...)` 报的位置是 **`for` 关键字**的位置（`N_FOR` 节点的 `tok`），不是循环体里出错语句的位置。这是有意的：死循环的责任在 `for` 头部，不在循环体。

**边界情况**：`loop_steps` 是**全局静态变量**，所有循环**共享**这一个预算。嵌套循环也会消耗同一个预算。程序正常跑完时它不会被重置——但这没关系，因为一个程序里只有一个 `main` 调用链。

**真实跑一遍**：写一个忘记写更新部分的 `for`：

```
let x = 0;
for (let i = 0; i < 3;) { x = x + 1; }
```

条件 `i < 3` 永远为真（`i` 永远是 0），更新部分又是空的 → 死循环。真实运行：

```sh
$ time ./minilang /tmp/infinite.ml
错误 [2:1]: 循环执行超过安全上限 1000000 步
$ echo $?
1
```

实测耗时约 **0.24 秒**（`time` 输出：`0.24s user 0.00s system 98% cpu 0.244 total`）。退出码 **1**。

注意报错位置 `[2:1]`——`for` 关键字在第 2 行第 1 列。

**如果你把安全阀去掉会怎样**：程序会一直跑下去，你只能 `Ctrl-C`。所以这个上限不是可有可无的。

### 5.6 概念六：更新部分目前只支持 `i = 表达式`

**问题**：`for (let i = 0; i < 3; i++)` 能写吗？

**思路**：**不能**。MiniLang 目前没有 `++` / `--` 运算符，词法分析器里也没有 `T_INC` / `T_DEC` 这两个 token（第 1 步的 token 表里就没有）。所以更新部分只接受"一个标识符 + `=` + 表达式"：

```c
if (cur()->type != T_RP) {
    Token id = expect(T_ID, "for 更新部分需要变量赋值");
    Node *x = node(N_ASSIGN, id);
    snprintf(x->text, sizeof(x->text), "%s", id.text);
    expect(T_ASSIGN, "for 更新部分需要 '='");
    x->a = expression();
    n->c = x;
}
```

真实运行：

```sh
$ cat /tmp/plusplus.ml
for (let i = 0; i < 3; i++) { print(i); }
$ ./minilang /tmp/plusplus.ml
错误 [1:25]: 语法错误：for 更新部分需要 '='，当前是 '++'
```

**这是一个明确的语言限制，不是 bug。** 想自增就写 `i = i + 1`。

**为什么不做 `i++`？** 因为 `i++` 的语义是"取值后加一"，在表达式里是个**有副作用**的运算，而 MiniLang 现在的表达式求值器（`eval`）是**纯函数式**的——它只算值，不改状态。加 `i++` 需要给 `eval` 加"副作用"通道，复杂度会跳一个台阶。而 `i = i + 1` 放在**语句**位置（更新部分），正好是 `N_ASSIGN` 节点，不需要 `eval` 变脏。**先做简单的那个。**

同理，更新部分也不能写 `let j = 1`：

```sh
$ ./minilang /tmp/let_update.ml
错误 [1:24]: 语法错误：for 更新部分需要变量赋值，当前是 'let'
```

### 5.7 概念七：初始化部分的两种写法

初始化部分可以是三种东西：

| 写法 | AST | 效果 |
|---|---|---|
| `for (let i = 0; ...)` | `N_LET` | 在**循环自己的作用域**里新建变量 `i` |
| `for (i = 0; ...)` | `N_ASSIGN` | 给**外层已存在**的变量 `i` 赋值 |
| `for (; ...)` | `NULL` | 什么都不做 |

**代码要点**：解析器用三路分支区分它们：

```c
if (match(T_SEMI)) {
    n->a = NULL;                       /* 留空 */
} else if (match(T_LET)) {
    ... 构造 N_LET ...
} else {
    ... 构造 N_ASSIGN ...
}
```

**边界情况**：`for (i = 0; ...)` 里的 `i` 必须在**外层已经定义过**，否则运行时报错：

```sh
$ ./minilang /tmp/undef_init.ml
错误 [1:6]: 不能给未定义变量 'i' 赋值
```

（列号 6 是 `i` 的位置——`for (i` 里 `(` 在第 5 列，`i` 在第 6 列。）

## 6. 动手：把代码加进来

### 6.1 改动清单

| 位置锚点 | 改了什么 | 大概多少行 | 为什么 |
|---|---|---|---|
| 宏定义区（`#define MAX_NAME 128` 之后） | 新增 `#define MAX_LOOP_STEPS 1000000` | 1 行 | 安全阀的上限值 |
| 全局变量区（`static int had_error;` 之后） | 新增 `static int loop_steps;` | 1 行 | 全局步数计数器 |
| `typedef enum { ... } NodeType;`（`N_IF` 之后） | 新增 `N_FOR` 枚举值 | 1 行 | AST 里要能表示"循环" |
| `statement()` 里，`if` 分支之后、`let` 分支之前 | 新增 `for` 的解析分支 | 46 行 | 把 `for (init; cond; upd) body` 变成 `N_FOR` 节点 |
| `dump_ast()` 的 `switch` 里，`case N_IF:` 之后 | 新增 `case N_FOR:` | 7 行 | `ast` 模式要打印出 `循环` 树 |
| `exec()` 的 `switch` 里，`case N_IF:` 之后 | 新增 `case N_FOR:` | 30 行 | `run` 模式要执行循环 |

合计 **86 行新增**，**0 行修改**（除枚举尾部加逗号）。

### 6.2 分段讲解

#### 第 1 段：安全阀的两个零件

```diff
 #define MAX_NAME   128
+#define MAX_LOOP_STEPS 1000000
```

```diff
 static int   had_error;          /* 出过错没有 */
+static int   loop_steps;         /* 循环一共执行了多少步（用来防死循环） */
```

**逐行解释**：

- 宏放在其它 `#define` 后面，保持"一类东西挨在一起"（第 1 步定型的 `MAX_TOKENS` / `MAX_NODES` / `MAX_TEXT` / `MAX_PARAMS` / `MAX_VARS` / `MAX_NAME`）。
- `loop_steps` 放在 `had_error` 后面——两者都是"全局状态"，`had_error` 是第 1 步的，`loop_steps` 是本步的。
- **注意 `loop_steps` 是 `static int`，初值是 0。** C 里静态存储期的变量会自动初始化为 0，不需要显式写 `= 0`。

**如果你写错了会怎样**：

- 把 `loop_steps` 写成局部变量（比如在 `case N_FOR:` 里声明）→ 每次进入循环都重置为 0，安全阀**永远不触发**，死循环会把解释器卡死。
- 把上限写成 `100000000`（一亿）→ 死循环要跑好几分钟才报错，用户会以为程序"挂了"。

#### 第 2 段：枚举加 `N_FOR`

```diff
     N_EXPR,     /* 表达式语句        */
-    N_IF        /* if (条件) ... else ... */
+    N_IF,       /* if (条件) ... else ... */
+    N_FOR       /* for (init; cond; update) { ... } */
 } NodeType;
```

**逐行解释**：同上一节的老规矩——补逗号、追加到末尾、不动已有值。

**如果你写错了会怎样**：忘记逗号 → 编译错误 `expected '}' before 'N_FOR'`。

#### 第 3 段：解析 `for`（本步最长的一段）

```diff
+    /* for (初始化; 条件; 更新) 语句
+       三段之间用分号隔开，整个 for 语句本身不需要分号结尾。 */
+    if (match(T_FOR)) {
+        Node *n = node(N_FOR, t);
+        expect(T_LP, "for 后需要 '('");
+
+        /* 第一段：初始化。可以是 let i = 0; 也可以是 i = 0; 也可以留空 */
+        if (match(T_SEMI)) {
+            n->a = NULL;
+        } else if (match(T_LET)) {
+            Node *x = node(N_LET, t);
+            Token id = expect(T_ID, "for 初始化需要变量名");
+            snprintf(x->text, sizeof(x->text), "%s", id.text);
+            expect(T_ASSIGN, "变量定义需要 '='");
+            x->a = expression();
+            expect(T_SEMI, "for 初始化后需要 ';'");
+            n->a = x;
+        } else {
+            Node *x = node(N_ASSIGN, *cur());
+            Token id = expect(T_ID, "for 初始化需要变量名");
+            snprintf(x->text, sizeof(x->text), "%s", id.text);
+            expect(T_ASSIGN, "赋值需要 '='");
+            x->a = expression();
+            expect(T_SEMI, "for 初始化后需要 ';'");
+            n->a = x;
+        }
+
+        /* 第二段：条件。留空表示永远为真 */
+        if (cur()->type != T_SEMI) n->b = expression();
+        expect(T_SEMI, "for 条件后需要 ';'");
+
+        /* 第三段：更新。目前只支持 i = 表达式 这样的赋值 */
+        if (cur()->type != T_RP) {
+            Token id = expect(T_ID, "for 更新部分需要变量赋值");
+            Node *x = node(N_ASSIGN, id);
+            snprintf(x->text, sizeof(x->text), "%s", id.text);
+            expect(T_ASSIGN, "for 更新部分需要 '='");
+            x->a = expression();
+            n->c = x;
+        }
+
+        expect(T_RP, "for 头部需要 ')'");
+        n->items[n->count++] = statement();   /* 循环体 */
+        return n;
+    }
```

**逐行解释**（按块看）：

**（a）头部开头**

```c
Node *n = node(N_FOR, t);
expect(T_LP, "for 后需要 '('");
```

`node(N_FOR, t)` 里 `t` 是 `statement()` 开头取的 `*cur()`，此时指向 `for` 关键字。**这个 token 决定了死循环报错时的行列号。**

**（b）初始化部分**

```c
if (match(T_SEMI)) {
    n->a = NULL;
} else if (match(T_LET)) {
    ...
} else {
    ...
}
```

三路分支的关键在于 `match()` 的"试探"语义：`match(T_SEMI)` 失败时 `pos` 不动，所以后面还能试 `match(T_LET)`。

- 第一路：第一个字符就是 `;` → 初始化留空，`n->a = NULL`（显式写出来，虽然 `memset` 已经保证它是 0，但写出来更清楚）。
- 第二路：`let` → 造一个 `N_LET` 节点。注意它**复用**了第 5 步 `statement()` 里 `let` 的那套写法（`expect(T_ID)` → `expect(T_ASSIGN)` → `expression()`），只是自己动手而没有调用 `statement()`——因为 `statement()` 会把结尾的分号也吃掉，而这里分号要单独处理。
- 第三路：其它情况 → 只能是"赋值"，造一个 `N_ASSIGN` 节点。

**（c）条件部分**

```c
if (cur()->type != T_SEMI) n->b = expression();
expect(T_SEMI, "for 条件后需要 ';'");
```

这里用的是"**看**当前 token 是不是 `;`"而不是 `match()`，因为我们**不**想吃掉这个分号——它要留给 `expect(T_SEMI)` 去报错或消费。用 `match(T_SEMI)` 会把分号吃掉，`expect` 就没法检查了。

条件留空（`for (;;)`）→ `n->b` 保持 `NULL` → 运行时 `!n->b` 为真 → 永远继续循环。

**（d）更新部分**

```c
if (cur()->type != T_RP) {
    Token id = expect(T_ID, "for 更新部分需要变量赋值");
    Node *x = node(N_ASSIGN, id);
    ...
}
```

同理，用"看是不是 `)`"来判断更新部分是否留空。留空时 `n->c` 是 `NULL`，运行时 `if (n->c)` 为假 → 不执行更新。

**这里有个 C 细节**：`expect(T_ID, ...)` 返回的是 `Token`（按值），而 `node(N_ASSIGN, id)` 用的是 `id` 这个**局部变量**。注意 `Node` 结构体里的 `tok` 是**值拷贝**，不是指针——所以 `id` 出了作用域也没关系。

**（e）收尾**

```c
expect(T_RP, "for 头部需要 ')'");
n->items[n->count++] = statement();   /* 循环体 */
return n;
```

循环体用 `statement()` 解析，所以既可以是 `{ ... }` 块，也可以是单条语句。

**如果你写错了会怎样**：

- 条件部分写成 `if (match(T_SEMI)) n->b = expression();` → 分号被吃掉，后面的 `expect(T_SEMI)` 会去要第二个分号，报出莫名其妙的错。
- 更新部分忘了 `expect(T_RP, ...)` → `for (let i=0;i<3;i=i+1 { ... }` 会在 `expression()` 里把 `{` 当成非法表达式，报 `语法错误：此处需要数字、变量、函数调用或括号表达式`。
- 初始化部分忘了 `expect(T_SEMI, ...)` → `for (let i = 0 i < 3; ...)` 会在 `expression()` 解析完 `0` 之后看到 `i`，然后 `expect(T_SEMI)` 报 `语法错误：for 初始化后需要 ';'，当前是 'i'`。

#### 第 4 段：打印 `for` 节点

```diff
+        case N_FOR:
+            printf("循环\n");
+            if (n->a) kids[m++] = n->a;   /* 初始化 */
+            if (n->b) kids[m++] = n->b;   /* 条件 */
+            if (n->c) kids[m++] = n->c;   /* 更新 */
+            for (int i = 0; i < n->count; i++) kids[m++] = n->items[i];  /* 循环体 */
+            break;
```

**逐行解释**：

- 三个 `if (n->a/b/c)` 判空——初始化、条件、更新都可能是 `NULL`。
- 最后一行用 `for` 循环把 `items[]` 里的循环体加进 `kids`。这里**没有**判空，因为循环体一定存在（`items[0]` 一定被赋值）。
- 注意这里的 `for` 循环变量也叫 `i`——和 MiniLang 源码里的 `i` 没关系，纯 C 代码。

**如果你写错了会怎样**：

- 忘了一个判空 → `dump_ast(NULL, ...)` 会被调用。`dump_ast` 开头有 `if (!n) return;` 所以不会崩，但树里会多出一行空的 `├── `。

#### 第 5 段：执行 `for`

```diff
+        case N_FOR: {
+            /* 循环自带一个作用域：i 只在循环里活着 */
+            Env local;
+            env_init(&local, e);
 
+            /* 第一段：初始化，只执行一次 */
+            if (n->a) {
+                f = exec(n->a, &local);
+                if (f.returned) { env_free(&local); return f; }
+            }
+
+            /* 第二段：条件，每次迭代前检查一次 */
+            while (!n->b || eval(n->b, &local).value != 0) {
+                /* 安全阀：防止 while(true) 之类的死循环把解释器卡死 */
+                if (++loop_steps > MAX_LOOP_STEPS) {
+                    env_free(&local);
+                    fatal_at(n->tok, "循环执行超过安全上限 %d 步", MAX_LOOP_STEPS);
+                }
+                f = exec(n->items[0], &local);          /* 循环体 */
+                if (f.returned) { env_free(&local); return f; }
+                if (n->c) {                               /* 第三段：更新 */
+                    f = exec(n->c, &local);
+                    if (f.returned) { env_free(&local); return f; }
+                }
+            }
+            env_free(&local);
+            return noflow();
+        }
```

**逐行解释**：

- `Env local; env_init(&local, e);` —— 建立循环自己的作用域，`local.parent = e`。**这一句是"for 自带作用域"的全部实现。**
- `if (n->a) { f = exec(n->a, &local); ... }` —— 初始化**只执行一次**，而且是在 `while` 之前。
- `while (!n->b || eval(n->b, &local).value != 0)` —— 这是整个 for 的核心：
  - `!n->b` 为真（条件留空）→ 短路，**不求值** `n->b`（它是 NULL，求值会崩）。
  - 否则求值 `n->b`，`!= 0` 就是"真"。
  - `||` 的短路求值是第 4 步 `eval` 里 `N_BINARY` 的能力，这里直接用上了。
- `if (++loop_steps > MAX_LOOP_STEPS)` —— 安全阀。注意它在**循环体之前**，所以每轮迭代都会检查一次。
- `f = exec(n->items[0], &local);` —— 执行循环体。`items[0]` 就是解析时存进去的循环体。
- `if (f.returned) { env_free(&local); return f; }` —— 检查 `Flow`。第 7 步还没有 `return` 语句，所以 `f.returned` 永远是 0；但**这段检查必须写**，因为第 11 步加 `return` 时就不用了……不，恰恰相反：第 11 步加 `return` 时**正好靠它**把返回值传出去。这就是 INV-3（`exec` 从第一天就返回 `Flow`）的兑现。
- `if (n->c) { f = exec(n->c, &local); ... }` —— 更新部分在**循环体之后**执行。
- `env_free(&local); return noflow();` —— 循环正常结束，释放循环的作用域。

**如果你写错了会怎样**：

- **把更新部分写在循环体之前** → 变成"先更新再执行"，第一次迭代就把 `i` 加了 1，输出的序列会整体错位。
- **忘了 `env_free(&local)`** → 每执行一个 `for` 就泄漏一块 `Env` 的变量表内存。循环本身泄漏还不明显，但如果 `for` 在一个长循环里，内存会一路涨。
- **把 `env_init(&local, e)` 写成 `env_init(&local, NULL)`** → 循环里的变量找不到外层的变量，`i < 10` 会报"未定义变量 'i'"。
- **安全阀放在循环体之后** → 死循环的最后一次迭代会先执行完循环体才检查，白做一次功。而且如果循环体里有 `print`，会多打印一次。

### 6.3 完整 diff

> **怎么用这段 diff**：推荐**照着敲**——每改一小段就编译一次，比一次性粘贴更容易记住。
> 如果想直接应用，把下面的内容存成 `step.patch`，然后在**上一步的目录**里执行：
>
> ```sh
> patch -p1 < ../007/step.patch
> ```
>
> 应用完和本目录的 `minilang.c` **逐字节一致**（可以 `diff` 验证）。

<details>
<summary>点开看完整 diff（真实 diff -u 输出）</summary>

```diff
--- a/minilang.c
+++ b/minilang.c
@@ -33,6 +33,7 @@
 #define MAX_PARAMS 32
 #define MAX_VARS   2048
 #define MAX_NAME   128
+#define MAX_LOOP_STEPS 1000000
 
 /* ------------------------------------------------------------------ */
 /* Token：词法分析器产出的最小单位                                     */
@@ -59,6 +60,7 @@
 static const char *src;          /* 整份源代码 */
 static int   src_line, src_col;  /* 扫描器当前所在的位置 */
 static int   had_error;          /* 出过错没有 */
+static int   loop_steps;         /* 循环一共执行了多少步（用来防死循环） */
 
 /* ------------------------------------------------------------------ */
 /* 错误处理：从第一步就定好名字和签名，后面不再改                       */
@@ -272,7 +274,8 @@
     N_LET,      /* let x = 1;        */
     N_ASSIGN,   /* x = 1;            */
     N_EXPR,     /* 表达式语句        */
-    N_IF        /* if (条件) ... else ... */
+    N_IF,       /* if (条件) ... else ... */
+    N_FOR       /* for (init; cond; update) { ... } */
 } NodeType;
 
 typedef struct Node Node;
@@ -464,6 +467,52 @@
         return n;
     }
 
+    /* for (初始化; 条件; 更新) 语句
+       三段之间用分号隔开，整个 for 语句本身不需要分号结尾。 */
+    if (match(T_FOR)) {
+        Node *n = node(N_FOR, t);
+        expect(T_LP, "for 后需要 '('");
+
+        /* 第一段：初始化。可以是 let i = 0; 也可以是 i = 0; 也可以留空 */
+        if (match(T_SEMI)) {
+            n->a = NULL;
+        } else if (match(T_LET)) {
+            Node *x = node(N_LET, t);
+            Token id = expect(T_ID, "for 初始化需要变量名");
+            snprintf(x->text, sizeof(x->text), "%s", id.text);
+            expect(T_ASSIGN, "变量定义需要 '='");
+            x->a = expression();
+            expect(T_SEMI, "for 初始化后需要 ';'");
+            n->a = x;
+        } else {
+            Node *x = node(N_ASSIGN, *cur());
+            Token id = expect(T_ID, "for 初始化需要变量名");
+            snprintf(x->text, sizeof(x->text), "%s", id.text);
+            expect(T_ASSIGN, "赋值需要 '='");
+            x->a = expression();
+            expect(T_SEMI, "for 初始化后需要 ';'");
+            n->a = x;
+        }
+
+        /* 第二段：条件。留空表示永远为真 */
+        if (cur()->type != T_SEMI) n->b = expression();
+        expect(T_SEMI, "for 条件后需要 ';'");
+
+        /* 第三段：更新。目前只支持 i = 表达式 这样的赋值 */
+        if (cur()->type != T_RP) {
+            Token id = expect(T_ID, "for 更新部分需要变量赋值");
+            Node *x = node(N_ASSIGN, id);
+            snprintf(x->text, sizeof(x->text), "%s", id.text);
+            expect(T_ASSIGN, "for 更新部分需要 '='");
+            x->a = expression();
+            n->c = x;
+        }
+
+        expect(T_RP, "for 头部需要 ')'");
+        n->items[n->count++] = statement();   /* 循环体 */
+        return n;
+    }
+
     /* let 变量 = 表达式; */
     if (match(T_LET)) {
         Node *n = node(N_LET, t);
@@ -561,6 +610,13 @@
             kids[m++] = n->b;   /* then 分支 */
             if (n->c) kids[m++] = n->c;   /* else 分支（可能没有） */
             break;
+        case N_FOR:
+            printf("循环\n");
+            if (n->a) kids[m++] = n->a;   /* 初始化 */
+            if (n->b) kids[m++] = n->b;   /* 条件 */
+            if (n->c) kids[m++] = n->c;   /* 更新 */
+            for (int i = 0; i < n->count; i++) kids[m++] = n->items[i];  /* 循环体 */
+            break;
     }
 
     for (int i = 0; i < m; i++) {
@@ -816,7 +872,36 @@
             return exec(v.value ? n->b : n->c, e);
             /* 没有 else 时 n->c 是 NULL，exec(NULL) 直接返回"什么都没发生" */
         }
+
+        case N_FOR: {
+            /* 循环自带一个作用域：i 只在循环里活着 */
+            Env local;
+            env_init(&local, e);
 
+            /* 第一段：初始化，只执行一次 */
+            if (n->a) {
+                f = exec(n->a, &local);
+                if (f.returned) { env_free(&local); return f; }
+            }
+
+            /* 第二段：条件，每次迭代前检查一次 */
+            while (!n->b || eval(n->b, &local).value != 0) {
+                /* 安全阀：防止 while(true) 之类的死循环把解释器卡死 */
+                if (++loop_steps > MAX_LOOP_STEPS) {
+                    env_free(&local);
+                    fatal_at(n->tok, "循环执行超过安全上限 %d 步", MAX_LOOP_STEPS);
+                }
+                f = exec(n->items[0], &local);          /* 循环体 */
+                if (f.returned) { env_free(&local); return f; }
+                if (n->c) {                               /* 第三段：更新 */
+                    f = exec(n->c, &local);
+                    if (f.returned) { env_free(&local); return f; }
+                }
+            }
+            env_free(&local);
+            return noflow();
+        }
+
         default:
             fatal_at(n->tok, "这里不是一条语句");
     }
```

</details>

> 和上一节一样，diff 里**没有**文件头的变化——所有步骤的 `minilang.c` 头注释一字不差。

### 6.4 插入位置说明

在 `006/minilang.c` 上手改的话，六处代码分别放在：

1. **宏**：在 `#define MAX_NAME   128` 那一行**下面**加 `#define MAX_LOOP_STEPS 1000000`。
2. **全局变量**：在 `static int   had_error;          /* 出过错没有 */` 那一行**下面**加 `static int   loop_steps;`。
3. **枚举**：在 `N_IF        /* if (条件) ... else ... */` 那一行下面（先给它补个逗号）加 `N_FOR`。
4. **解析**：`static Node *statement(void)` 里，在**整个 `if` 分支结束之后**（也就是 `return n;` 和 `}` 之后）、`/* let 变量 = 表达式; */` 注释之前，插入 `if (match(T_FOR)) { ... }` 整块。
5. **打印**：`dump_ast()` 的 `switch` 里，在 `case N_IF:` 的 `break;` 之后。
6. **执行**：`exec()` 的 `switch` 里，在 `case N_IF: { ... }` 整块之后、`default:` 之前。

改完后编译：

```sh
cc -std=c11 -Wall -Wextra -O2 minilang.c -lm -o minilang
```

**编译器警告检查**：本步新增的代码里有几个容易触发 `-Wall -Wextra` 警告的地方：

- `case N_FOR: {` 里的 `Env local;` —— 必须有花括号，否则 `case` 之间会跨声明冲突。
- `f = exec(...)` 之后 `if (f.returned)` —— `f` 在 `case N_FOR:` 外面的 `Flow f = noflow();` 已经声明过了，本步只是赋值，不会有 "unused variable" 警告。
- 循环体里的 `for (int i = 0; ...)` —— C99 允许在 `for` 的初始化部分声明变量，C11 更是标配。

## 7. 测试

`./run.sh` 的真实输出：

```
▶ 用例（模式：run）
  ✅ case_01_sum.ml               [run]
  ✅ case_02_empty_init.ml        [run]
  ✅ case_03_no_update.ml         [run]
  ✅ case_04_loop_var_scope.ml    [run]
▶ demo.ml
  | 55
✅ 全部通过（5 项）
```

逐个用例说明：

**`tests/case_01_sum.ml`** —— 累加 1~5：

```
let total = 0;
for (let i = 1; i <= 5; i = i + 1) {
    total = total + i;
}
print(total);
```

- 期望 stdout：`15`（一个字符 `1` 和一个字符 `5`，**没有换行**）。
- 期望 stderr：空。期望退出码：0。
- 逐步推导：`i` 依次取 1、2、3、4、5，`total` 依次变成 1、3、6、10、15。第 6 轮检查 `6 <= 5` 为假 → 退出循环 → `print(total)` 输出 `15`。
- 注意 `total` 是**外层**变量，而循环体是一个块——块里的 `total = total + i` 是**赋值**，会沿作用域链往外找到外层的 `total` 并修改它。（如果是 `let total = ...` 就会新建一个局部变量，外面的 `total` 永远是 0。）

**`tests/case_02_empty_init.ml`** —— 初始化留空：

```
let i = 0;
for (; i < 3; i = i + 1) { print(i); }
```

- 期望 stdout：`012`（三个字符，**没有换行**）。
- 期望 stderr：空。期望退出码：0。
- 逐步推导：初始化留空 → `n->a = NULL` → 什么都不做。`i` 用的是**外层**那个（第 1 行定义的）。第 1 轮 `i = 0` → 打印 `0` → 更新 `i = 1` → 第 2 轮打印 `1` → 更新 `i = 2` → 第 3 轮打印 `2` → 更新 `i = 3` → 第 4 轮 `3 < 3` 为假 → 退出。
- 这个用例验证的是**初始化留空的解析路径**（`if (match(T_SEMI)) n->a = NULL;` 那一支）。

**`tests/case_03_no_update.ml`** —— 死循环 + 安全阀：

```
let x = 0;
for (let i = 0; i < 3;) { x = x + 1; }
```

- 期望 stderr（全文，一行）：

  ```
  错误 [2:1]: 循环执行超过安全上限 1000000 步
  ```

- 期望 stdout：空。期望退出码：**1**。
- 为什么是 `[2:1]`：`for` 关键字在第 2 行第 1 列。`fatal_at(n->tok, ...)` 里的 `n` 是 `N_FOR` 节点，它的 `tok` 是解析 `statement()` 时取的 `*cur()`——也就是 `for` 那个 token。
- 为什么不报错"变量 'x' 未使用"之类：`x = x + 1` 一直在赋值，`x` 早就定义过了，不会报错。循环只是永远跑下去。
- **实测耗时约 0.24 秒**——100 万次迭代，每次迭代要做一次条件求值 + 一次块执行。

**`tests/case_04_loop_var_scope.ml`** —— 循环变量的作用域：

```
for (let i = 0; i < 2; i = i + 1) { print(i); }
print(i);
```

- 期望 stdout：`01`（**注意：报错之前已经打印了 `01`**）。
- 期望 stderr（全文，一行）：

  ```
  错误 [2:7]: 未定义变量 'i'
  ```

- 期望退出码：**1**。
- 为什么是 `[2:7]`：第 2 行是 `print(i);`——`print` 占 1~5 列，`(` 占第 6 列，`i` 在第 7 列。
- 为什么先输出 `01` 再报错：程序是**顺序执行**的，第 1 行的循环先跑完并打印了 `0` 和 `1`，第 2 行才去找 `i`，此时循环已经结束、`local` 环境已经被 `env_free` 释放，`i` 不存在了。
- **这个用例同时验证了两件事**：循环体里能看见 `i`（`01` 打印出来了），循环外看不见 `i`（报未定义变量）。

**`demo.ml` 冒烟测试** —— 输出 `55`（1+2+...+10 = 55）。

## 8. 易错点与坑

### 坑 1：在 `for` 后面多写一个分号

- **错误现象**：`错误 [行:列]: 语法错误：此处需要数字、变量、函数调用或括号表达式`。
- **原因**：MiniLang **没有空语句**。C 里 `for (...) { };` 后面那个 `;` 是合法的空语句，MiniLang 的 `statement()` 不认识它，于是把它当成"表达式语句"去解析 `expression()`，而 `expression()` 看到 `;` 直接报错。
- **正确写法**：`for (...) { }` 后面什么都不写。**记住这个区别**：头部的两个分号是**必需的**，语句末尾的分号是**不能有的**。

### 坑 2：把头部分号和语句分号搞混

- **错误现象**：`for (let i = 0 i < 3; i = i + 1) { }` 报 `语法错误：for 初始化后需要 ';'`。
- **原因**：C 语言的 `for` 头部有**两个**分号，把头部分成三段。MiniLang 沿用这个约定。
- **正确写法**：`for (let i = 0; i < 3; i = i + 1) { }`。三个部分之间的分号不能少，末尾 `)` 之前**不要**再加分号。

### 坑 3：以为 `i++` 能用

- **错误现象**：`错误 [行:列]: 语法错误：for 更新部分需要 '='，当前是 '++'`。
- **原因**：MiniLang 没有 `++` / `--` 运算符（第 1 步的 token 表里就没有 `T_INC`），更新部分只接受"标识符 = 表达式"。
- **正确写法**：`i = i + 1`。这是**语言限制**，不是 bug。

### 坑 4：安全阀写成局部变量

- **错误现象**：死循环把解释器卡死，怎么等都不出来。
- **原因**：如果你把 `loop_steps` 声明在 `case N_FOR:` 里面（比如 `int loop_steps = 0;`），每次进入循环都会重置为 0，`++loop_steps > MAX_LOOP_STEPS` 永远不成立。
- **正确写法**：`loop_steps` 必须是**文件级**的 `static int`，和 `had_error` 放在一起。

### 坑 5：忘了 `env_free(&local)`，或者把 `parent` 写错

- **错误现象（前者）**：反复执行 `for` 会让内存一路上涨。程序功能上没差别，但这是泄漏。
- **错误现象（后者）**：`env_init(&local, e)` 写成 `env_init(&local, NULL)` → 循环里报"未定义变量 'i'"，因为 `i` 定义在 `local` 里但查找链断了。
- **正确写法**：`env_init(&local, e);` … `env_free(&local);` 成对出现。

### 坑 6：`exec` 的 `case` 里忘了写花括号

- **错误现象**：编译报错 `a label can only be part of a statement and a declaration is not a statement` 或 `crosses initialization of 'Env local'`。
- **原因**：`case N_FOR: { Env local; ... }` —— C 的 `switch` 里所有 `case` 共享同一个块作用域，`case` 里直接声明变量会跨 case 冲突。
- **正确写法**：给 `case` 加上 `{ }`，注意结尾也要有 `}`。

### 坑 7：把更新部分写在循环体之前

- **错误现象**：输出的序列整体错位，比如本该打印 `0 1 2`，实际打印 `1 2 3`。
- **原因**：C 的 `for` 语义是"先判断条件，再执行循环体，最后更新"。把更新提到循环体前面，等于每轮先把 `i` 加一。
- **正确写法**：对着 5.2 节的展开图写代码：

  ```
  init;
  while (cond) {
      BODY;
      update;      ← 更新在最后
  }
  ```

### 坑 8：以为 `loop_steps` 是每个循环独立的预算

- **错误现象**：写一个外层 1000 次、内层 1000 次的嵌套循环（总共 100 万次迭代），结果在**内层**就报"超过安全上限"。
- **原因**：`loop_steps` 是全局的，所有循环共享。
- **正确写法**：知道这一点就行。如果需要更精细的控制，可以把计数器改成"每条 for 语句一个"，但那需要在 AST 节点里加字段（违背 INV-5），或者用 `n->count` 之类的字段临时存——不值得。

## 9. 自测题

### 9.1 概念题

**Q1.** `for` 的三个部分分别对应 AST 的哪几个字段？循环体呢？

**Q2.** 为什么 `for` 语句本身不需要分号结尾，而头部里却必须要两个分号？

**Q3.** `for (let i = 0; i < 3; i = i + 1) { print(i); }` 循环结束后，`i` 去哪了？

**Q4.** 为什么 `loop_steps` 必须是全局变量，不能是局部变量？

**Q5.** 在 `exec` 里，初始化、条件、更新、循环体分别在什么时候执行？各自执行几次？

<details>
<summary>参考答案</summary>

**A1.** 初始化 → `n->a`（`N_LET` 或 `N_ASSIGN`，可以为空）；条件 → `n->b`（表达式节点，可以为空）；更新 → `n->c`（`N_ASSIGN`，可以为空）；循环体 → `n->items[0]`（语句节点，不能为空）。

**A2.** 因为 C 式 `for` 的头部本身就用分号把三段分开来了，两个分号是头部的语法的一部分；而 `for` 整体是一条语句，C 语言里语句末尾也不需要额外的分号。MiniLang 没有空语句，所以多写一个分号反而是语法错误。

**A3.** `i` 定义在 `for` 自己的 `Env`（`local`）里。循环结束时 `env_free(&local)` 把这个环境销毁了，`i` 随之消失。循环外再引用 `i` 会报"未定义变量 'i'"。

**A4.** 因为计数器要跨循环累加。局部变量每次进入 `case N_FOR:` 都会被重新初始化为 0，`++loop_steps > MAX_LOOP_STEPS` 永远不成立，死循环就卡死了。

**A5.** 初始化：在 `while` 之前，执行**一次**（或 0 次，如果留空）。条件：每次迭代**开始前**检查一次，共 N+1 次（N = 实际迭代次数）。循环体：每轮一次，共 N 次。更新：每轮循环体**之后**一次，共 N 次（或 0 次，如果留空）。

</details>

### 9.2 预测题（"改一行会发生什么"）

**Q6.** 把 `exec` 里的 `while (!n->b || eval(n->b, &local).value != 0)` 改成 `while (1)`，`tests/case_01_sum.ml` 会怎样？

**Q7.** 把更新部分的执行挪到循环体**之前**（即把 `if (n->c) {...}` 移到 `f = exec(n->items[0], &local);` 前面），`tests/case_01_sum.ml` 的输出会变成什么？

**Q8.** 把 `n->items[n->count++] = statement();` 改成 `n->b = statement();`（循环体放到 `b` 里），`tests/case_01_sum.ml` 的输出会变成什么？

**Q9.** 把 `if (++loop_steps > MAX_LOOP_STEPS)` 改成 `if (++loop_steps > 10)`，`tests/case_03_no_update.ml` 的 stderr 会变成什么？

<details>
<summary>参考答案</summary>

**A6.** `tests/case_01_sum.ml` 不再输出 `15`，而是报 `错误 [3:1]: 循环执行超过安全上限 1000000 步`（`for` 在第 3 行第 1 列）并以退出码 1 结束。因为循环条件被写死成"永远为真"，安全阀会在 100 万步后拦下来。

**注意 `!n->b ||` 这一小段唯一的作用就是支持"条件留空"**：去掉它以后，`for (;;)` 里的 `n->b` 是 `NULL`，`eval(NULL, ...)` 会返回 `number(0)`（`eval` 开头有 `if (!n) return number(0);`），`0 != 0` 为假 → 循环一次都不执行。想验证这一点，写一个 `for (;;) { print(1); }` 跑一跑就行。

**A7.** 输出变成 `20`。推导（每一轮都是"先更新，再执行循环体"）：

| 轮次 | 检查条件时的 `i` | 更新后 `i` | 累加的数 | `total` |
|---|---|---|---|---|
| 1 | 1 | 2 | 2 | 2 |
| 2 | 2 | 3 | 3 | 5 |
| 3 | 3 | 4 | 4 | 9 |
| 4 | 4 | 5 | 5 | 14 |
| 5 | 5 | 6 | 6 | 20 |

第 6 轮检查时 `i = 6`，`6 <= 5` 为假 → 退出 → `print(total)` 输出 `20`。

**A8.** 条件部分会写进 `n->b`，而更新部分也会写进 `n->c`——但 `n->b` 原本装的是条件！所以条件节点被循环体覆盖 → `eval(n->b)` 求值的是**循环体**（一个 `N_BLOCK`）→ `eval` 遇到语句节点 → `fatal_at(n->tok, "此处不是表达式")`。程序会报错退出，而不是输出 `15`。

**A9.** stderr 变成：

```
错误 [2:1]: 循环执行超过安全上限 10 步
```

因为 `%d` 代入的是 `MAX_LOOP_STEPS` 的值——你把宏改成了 10，报错文本里的数字也跟着变（`fatal_at` 用的是 `%d` 格式 + 宏本身）。所以**宏和错误文本是绑在一起的**。

</details>

### 9.3 动手题（改代码 + 跑 tests 验证）

**Q10.** 写一个 `.ml` 程序，用**嵌套 `for`** 打印九九乘法表的前 3 行（提示：需要两个 `for`，以及一个能算乘积的表达式）。注意：第 7 步的 MiniLang **还不能输出空格和换行**——`print` 只会打印数字，所以你想不出怎么分隔，就用 `print(100)` 打一个"分隔标记"。

**Q11.** 验证 `for` 的三种初始化写法。请分别写三个 `.ml` 并用 `./minilang` 跑出真实输出：

- `a.ml`：`for (let i = 0; i < 2; i = i + 1) { print(i); }` → 期望 `01`
- `b.ml`：`let i = 9; for (i = 0; i < 2; i = i + 1) { print(i); } print(i);` → 期望 `012`（说明赋值改的是外层的 `i`）
- `c.ml`：`for (; 1; ) { print(1); }` → 期望命中安全阀报错

**Q12.**（进阶）给 MiniLang 加一个 `while` 语句：`while (条件) { ... }`。提示：

- 加一个 `N_WHILE` 枚举值。
- 解析：`expect(T_LP)` → `n->a = expression()` → `expect(T_RP)` → `n->b = statement()`。
- 执行：和 `for` 的循环部分几乎一样——`while (eval(n->a, &local).value != 0) { 安全阀; 执行 n->b; }`。
- 别忘了 `dump_ast` 里加 `case N_WHILE:`。

改完后验证：`while (1) { }` 会不会也命中安全阀？

<details>
<summary>参考答案</summary>

**A10.** 一个能跑通的版本：

```
for (let i = 1; i <= 3; i = i + 1) {
    for (let j = 1; j <= i; j = j + 1) {
        print(j);
        print(i);
        print(i * j);
        print(100);
    }
}
```

真实输出：

```sh
$ ./minilang /tmp/table.ml
111100122100224100133100236100339100
```

（`100` 是人为选的分隔标记——第 7 步的 `print` 还不会输出字符串，所以没法打空格和换行，只能用一个不会和别的数字混淆的数当分隔符。第 9 步的 `print` 模板会彻底解决这个问题。）

**为什么内层循环是 `j <= i` 而不是 `j <= 3`**：这样打印出来的才是"三角形的乘法表"（第 1 行 1 项、第 2 行 2 项、第 3 行 3 项）。把 `j <= i` 改成 `j <= 3` 会打印出一个 3×3 的方阵。

**A11.** 三份文件的真实运行结果：

```
$ ./minilang a.ml
01
$ ./minilang b.ml
012
$ ./minilang c.ml
错误 [1:1]: 循环执行超过安全上限 1000000 步
```

`b.ml` 的 `012` 解释了作用域链：更新部分 `i = i + 1` 是**赋值**，`set_var` 从 `local` 往外找，在外层找到 `i` 并修改它，所以循环结束后外层的 `i` 是 2。

`c.ml` 的报错位置是 `[1:1]`——`for` 关键字在第 1 行第 1 列。`c.ml` 的初始化留空、条件是 `1`（永远为真）、更新也留空，所以它是一个纯死循环，100 万步后被安全阀拦下。

**A12.** 加了 `while` 之后：

```
while (1) { }
```

会命中安全阀，报 `错误 [1:1]: 循环执行超过安全上限 1000000 步`——因为 `while` 的执行代码和 `for` 的循环部分共用同一个 `loop_steps` 计数器。实现的时候记得：

- `N_WHILE` 节点的条件放 `n->a`，循环体放 `n->b`（或者也放 `items[0]`，保持和 `for` 一致）。
- `while` **也**应该自带作用域（`Env local`），和 `for` 一样。
- `while` 的条件留空要报错（C 语言的 `while` 必须有条件），所以解析时直接 `expect` 一个表达式。

</details>

## 10. 本步的设计取舍

**为什么用 C 式 `for` 而不是 `while` 或者更简单的 `repeat n times`？**

因为读者的目标是"学会写解释器"，不是"学一门新语言"。C 式 `for` 是所有人都会写的，而且它的三段结构正好对应 AST 的三个字段——**语法结构和数据结构对齐**是最重要的教学价值。相比之下 `repeat 10 { }` 只需要一个字段，反而学不到"一个节点有多个子节点"这件事。

**为什么循环体放在 `items[0]` 而不是 `d`？**

`Node` 结构体是第 2 步定型的（INV-5），加字段会破坏它。`items[128]` 本来就是"一串子节点"的通用容器（第 2 步给 `N_CALL` 的实参、第 5 步给 `N_BLOCK` 的语句），循环体虽然只有一个，但语义上它就是"for 节点的孩子列表"。**复用比新增更好。**

**为什么循环自带作用域？**

因为 C 的 `for` 也自带作用域：`for (int i = 0; ...)` 的 `i` 在循环外不可见。照搬 C 的行为，读者才不会写出"我以为 `i` 循环后还在"的错误程序。代价是：想在循环外用 `i`，得在循环外面先 `let i = 0;`（就像 `tests/case_02_empty_init.ml` 那样）。

**安全阀放在全局，为什么不是一个"每个 for 独立的预算"？**

因为需要一个跨循环累加的全局计数器最简单。全局预算的缺点是嵌套循环会共享（8. 坑 8）。更精细的做法（比如每个 for 语句单独计数）需要在 AST 节点里加一个 int 字段——那会破坏 INV-5，或者要 hack `n->count`（但 `count` 已经被 `items` 用了）。**这个取舍是"简单够用"优先。**

**还没做的：**

- 没有 `while` 语句（第 9.3 节的 Q12 是个不错的练习）。
- 没有 `i++` / `i--`。
- 没有 `break` / `continue`——想提前退出循环，只能把条件写成 `i < 5 && x < 100` 这种复合条件。
- 没有 `for (let i = 0; i < n; i = i + 1)` 之外的迭代形式（没有 `for x in list`）。
- 死循环是靠步数上限兜底的，不是靠 `break`。
- 步数上限是**写死的宏**，不能通过命令行调。

## 11. 下一步预告

现在 MiniLang 已经能算、能分支、能循环了，但还缺一样最基础的东西：**数学函数**。你想写 `sqrt(16)` 而不是 `16` 的平方根？想让程序算绝对值、幂、取整？下一节我们给 MiniLang 加一组内置函数：`abs` / `round` / `sqrt` / `floor` / `ceil` / `pow` / `min` / `max`。

你会看到为什么这时候要引入**表驱动**的实现方式（一张 `builtins[]` 表 + 一个查表函数），而不是写一长串 `if (strcmp(name, "abs") == 0) ...`；也会看到 C 里函数指针的语法 `double (*fn)(double *args, Token tok)` 到底该怎么读；还有为什么 `print` 明明也是内置函数，却**不能**放进这张表里。

下一节：[第 8 步 · 内置函数](../008/README.md)
