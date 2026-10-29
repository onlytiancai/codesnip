# 第 6 步 · if / else 条件分支

> 进度 ██████░░░░░░ 6/13 ｜ 上一节：[第 5 步 · 语句 / let / 作用域](../005/README.md) ｜ 下一节：[第 7 步 · for 循环](../007/README.md)

## 1. 本步目标

读完你将能够：

- 让程序按条件选择分支：`if (x > 5) { print(1); } else { print(0); }` 在 `x = 10` 时输出 `1`。
- 写出只有 `if`、没有 `else` 的条件语句，条件为假时**什么都不做**（不报错、不输出）。
- 理解 MiniLang 的条件是"一条语句"而不是"一个块"：`if (x) let y = 1;` 语法合法，但 `y` 会泄漏到外层作用域。
- 解释 `0` 是假、`非 0` 是真这条真值规则，并说出 `-0.5` 为什么是"真"。
- 用 `ast` 模式看懂一棵带 `if` 的语法树长什么样。

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
| `tokens` | 任意 `.ml` | token 列表（`IF` / `ELSE` / `LP` / `RP` 都在里面） | 0（有错为 1） |
| `ast` | 任意 `.ml` | 语法树，`if` 打印成 `如果` 节点 | 0（有错为 1） |
| `run`（默认） | 任意 `.ml` | 真正执行程序，分支只执行被选中的那一支 | 0；语法/运行错误为 1 |

第 5 步已经支持 `let`、赋值、代码块；本步**只增加**了条件分支，其他语法一行没动。

`006/demo.ml` 的内容：

```
// 第 6 步：条件分支
let x = 10;
if (x > 5) {
    print(1);
} else {
    print(0);
}
```

真实运行结果：

```sh
$ ./minilang demo.ml
1
```

用 `ast` 模式看这棵树：

```sh
$ ./minilang demo.ml ast
语法树：
代码块
├── 定义 x
│   └── 数字 10
└── 如果
    ├── 二元 GT
    │   ├── 变量 x
    │   └── 数字 5
    ├── 代码块
    │   └── 表达式语句
    │       └── 调用 print
    │           └── 数字 1
    └── 代码块
        └── 表达式语句
            └── 调用 print
                └── 数字 0
```

注意 `如果` 节点下面有 **3 个** 子节点：条件、then 分支、else 分支。如果源码里没写 `else`，那么第三个子节点**根本不存在**——打印时也就少一行。

## 4. 前置知识回顾

上一步我们已经在 `minilang.c` 里搭好了三块地基，本步全部要用上：

**（1）`statement()` 是"一条语句"的入口。** 第 5 步引入的语句层把程序拆成"语句 → 块 / let / 赋值 / 表达式语句"四种。本步在 `statement()` 里再加一种：`if`。注意 `statement()` 这个函数返回的是**一个** `Node *`，不是列表——所以"一条语句"这个概念是贯穿整套实现的。

**（2）`exec()` 执行语句，`eval()` 执行表达式，两者物理分离。** 这条约定（INV-4）到本步还在生效：`if` 的**条件**走 `eval()`，两个**分支**走 `exec()`。看到 `eval`/`exec` 的区别，就知道它算的是"值"还是"动作"。

**（3）`Env` 作用域链和 `exec(NULL)` 的语义。** 第 5 步引入的作用域（`Env` 有 `parent` 指针，找不到变量就沿链往外找）是本步"分支是一条语句"这个设计的直接后果：分支里写 `let`，变量会定义在**当前**环境里。至于"没有 else 时怎么办"，答案就藏在 `exec` 的第一行——见第 5 节。

## 5. 本步引入的核心概念

### 5.1 概念一：条件语句的"三条腿"——条件 / then / else

**问题**：C 里的 `if` 有两个子结构（条件、分支体），`else` 又算第三个。MiniLang 的 AST 节点结构体是固定的：

```c
struct Node {
    NodeType type;
    Token   tok;
    double  num;
    char    text[MAX_TEXT];
    int     op;
    Node   *a, *b, *c;        /* 子节点 */
    Node   *items[128];
    int     count;
    char   *params[MAX_PARAMS];
    int     nparams;
};
```

只有 `a` / `b` / `c` 三个通用子指针和一个 `items[]` 数组。加新的语句类型时，**不动结构体**（这正是 INV-5 "表结构一步定型"的意思），只把新的子节点塞进这三个指针里。

**思路**：`N_IF` 就规定 `a` = 条件，`b` = then 分支，`c` = else 分支：

```
              N_IF
             / | | \
            a  b  c
         条件 then else
                (可为 NULL)
```

对应到内存里就是这样：

```c
n->a = expression();          /* 条件：一条表达式 */
n->b = statement();           /* then 分支：一条语句 */
if (match(T_ELSE)) n->c = statement();   /* else 分支：可能没有 */
```

**代码要点**：`node()` 工厂函数里有一句 `memset(n, 0, sizeof(*n));`。这句至关重要——因为 `n->c` 只有在遇到 `else` 时才会被赋值。没有 `else` 时，`n->c` 保持 `memset` 后的 **0**，也就是 `NULL`。**你没有写 `n->c = NULL;`，但它确实是 NULL。**

**边界情况**：`dump_ast()` 打印时必须判断 `if (n->c)`，否则会把一个 NULL 指针传给 `dump_ast`。好在 `dump_ast` 的第一行是 `if (!n) return;`，所以即使传 NULL 也不会崩——但那样打印出来的树会多一个空节点，读者会困惑。所以两处都要照顾到。

### 5.2 概念二：分支是"一条语句"，不是"一个块"

**问题**：`if (x) { ... }` 和 `if (x) let y = 1;` 在语法分析器看来有什么不同？

**思路**：**没有不同**。`statement()` 里 `n->b = statement();` —— then 分支调用的还是 `statement()`，它既可能返回 `N_BLOCK`（`{ ... }`），也可能返回 `N_LET`（`let y = 1;`）、`N_ASSIGN`（`y = 1;`）、`N_EXPR`（`f();`）、甚至嵌套的 `N_IF`。

这和 C 完全一致：C 语言里 `if (x) let y = 1;` 这种写法在 C 里不合法（因为 C 的 `if` 后必须跟**语句**，而 `let` 不是 C 语句），但 C 里的 `if (x) y = 1;` 是合法的——`y = 1;` 本身就是一条语句。

```
MiniLang 语法：
   if (条件) 语句
   if (条件) 语句 else 语句
             ↑
        一条语句！可能是块，也可能只是单条

所以这两种都合法：
   if (x) { print(1); print(2); }   ← 块：两条语句
   if (x) print(1);                 ← 单条语句
```

**代码要点**：想写多条语句，**必须**用 `{ }` 包起来。

**边界情况（重要！）**：不用 `{ }` 的时候，作用域会出问题：

```
let x = 1;
if (x) let y = 1;
print(y);
```

这段代码语法上完全合法，而且**能跑通**——输出 `1`。原因是 `exec` 里 `case N_LET:` 调的是 `define_var(e, ...)`，而这里的 `e` 就是**外层**环境（没有块就没有新环境）。也就是说 `y` 被定义到了全局作用域，`if` 外面也能看见。

真实运行：

```sh
$ cat /tmp/demo_if_let.ml
let x = 1;
if (x) let y = 1;
print(y);
$ ./minilang /tmp/demo_if_let.ml
1
```

这不是 bug，是"分支是一条语句"的必然结果。**养成永远写 `{ }` 的习惯。**

### 5.3 概念三：`exec(NULL)` —— "什么都没发生"

**问题**：条件为假、又没有 `else` 的时候，要执行什么？

**思路**：什么都不执行。这在 C 里是"空语句"的意思。在 MiniLang 里，我们不引入空语句节点，而是让 `exec()` 支持接收 `NULL`：

```c
static Flow exec(Node *n, Env *e) {
    if (!n) return noflow();      /* ← 这一行就是为此存在的 */
    ...
}
```

于是 `N_IF` 的执行就可以写得非常干净：

```c
return exec(v.value ? n->b : n->c, e);
```

条件为真 → `exec(n->b)`；条件为假 → `exec(n->c)`，而 `n->c` 是 `NULL` → `exec` 直接返回 `noflow()`（"什么都没发生"）。

```
        条件为真                      条件为假
   ┌──────────────┐            ┌──────────────┐
   │ exec(n->b)   │            │ exec(n->c)   │
   │  执行 then   │            │  n->c == NULL │
   │              │            │  → noflow()  │
   └──────────────┘            └──────────────┘
```

**代码要点**：`noflow()` 返回的是 `Flow { .returned = 0, .value = number(0) }`，意思是"这条语句链没有提前 return"。这一点在第 11 步（return 与递归）会变得非常重要——那时 `Flow` 要沿着调用链往上传。

**边界情况**：`exec(NULL)` 只能处理"不执行"，不能处理"执行空语句"。`if (x) ;`（后面跟一个分号）在 MiniLang 里是**语法错误**，因为 `statement()` 不认识空语句：

```sh
$ ./minilang /tmp/empty_stmt.ml
错误 [1:7]: 语法错误：此处需要数字、变量、函数调用或括号表达式
```

### 5.4 概念四：条件的求值规则

**问题**：什么算"真"？条件表达式求出来是个 `Value`（第 3 步定型的 `{ double value; int is_string; char string[]; }`）。

**思路**：MiniLang 沿用 C 的真值规则：

| 条件的值 | 真假 |
|---|---|
| `0` | 假 |
| `0.0` | 假 |
| `非 0`（包括 `-0.5`、`0.5`、`3`） | 真 |
| 字符串 | 报错（见下） |

判断代码只有一句：`v.value ? n->b : n->c`。C 里 `0` 是假、其它非零值是真，这里完全一致。

真实运行：

```
if (0) { print(1); } else { print(2); }     →  2
if (-0.5) { print(3); }                     →  3
if (0.0) { print(4); } else { print(5); }   →  5
```

输出 `235`。

**关于字符串条件**：`exec` 里有一行 `if (v.is_string) fatal_at(n->tok, "if 的条件必须是数字或比较表达式");`。但在第 6 步，这行**根本走不到**——因为 `eval()` 遇到字符串字面量（`N_STR`）时就会先报错退出：

```sh
$ cat /tmp/str_cond.ml
if ("hi") { print(1); }
$ ./minilang /tmp/str_cond.ml
错误 [1:5]: 求值器还不支持字符串，字符串会在讲 print 模板时支持
```

注意报错的行列是 **`[1:5]`**，也就是字符串字面量自己的位置，而不是 `if` 的位置（`[1:1]`）。因为 `N_STR` 的 `tok` 是字符串 token，而 `N_IF` 的 `tok` 是 `if` token。

那 `exec` 里的字符串检查是干嘛的？**给第 9 步留的。** 第 9 步 `eval` 真的会返回字符串值（`print("hi {x}\n")` 里的模板求值），那时候 `if ("hi")` 就该报"条件必须是数字"了。这正是 INV-2（`Value` 里的字符串字段从第一天就预留）兑现价值的地方——**当时不写，第 9 步就得回头改 `exec` 的签名**。

### 5.5 概念五：条件只求值一次

**问题**：`if (条件)` 里的条件会被执行几次？

**思路**：**一次**。`exec` 里先 `Value v = eval(n->a, e);` 把值算出来存进局部变量 `v`，再用 `v.value` 去决定走哪条分支。条件表达式**不会**在选分支的时候被再执行一遍。

怎么验证？用一个有副作用的表达式。`print` 是目前唯一有副作用的函数，它打印完返回 `0`：

```
if (print(9) || 1) { print(1); }
```

- `print(9)` 执行一次，打印 `9`，返回 `0`。
- `0 || 1` → 短路求值（`||` 左边为假才看右边）→ 结果 `1`（真）。
- 进入 then 分支，打印 `1`。

真实输出：

```sh
$ ./minilang /tmp/side_effect.ml
91
```

如果条件被求值两次，你会看到 `991`。

### 5.6 概念六：悬垂 else（dangling else）

**问题**：

```
if (a) if (b) print(1); else print(2);
```

`else` 属于哪个 `if`？

**思路**：属于**最近的那个** `if`（即内层）。这和 C、Java、JavaScript 全都一致。规则来自"分支是一条语句"：`else` 跟在 `if (b) print(1);` 这条完整语句后面。

```
if (a)
└── if (b) print(1);
    else print(2);     ← else 属于内层 if
```

真实运行（`a = 1`, `b = 0`）：

```sh
$ ./minilang /tmp/dangling.ml
23
```

`a` 为真 → 进入外层 then → 内层 `if (b)`，`b = 0` 为假 → 执行内层的 `else` → 打印 `2`；接着外层的 then 分支结束，程序继续打印 `3`。输出 `23`。

**如果想给外层配 else**，必须用 `{ }` 把内层包起来：

```
if (a) { if (b) print(1); } else print(2);
```

## 6. 动手：把代码加进来

### 6.1 改动清单

| 位置锚点 | 改了什么 | 大概多少行 | 为什么 |
|---|---|---|---|
| `typedef enum { ... } NodeType;`（在 `N_EXPR` 后面） | 新增 `N_IF` 枚举值 | 1 行 | AST 里要能表示"如果"这种语句 |
| `statement()` 函数里，裸代码块 `if (cur()->type == T_LB) return block();` 之后 | 新增 `if` 的解析分支 | 11 行 | 把 `if (cond) stmt [else stmt]` 的 token 序列变成 `N_IF` 节点 |
| `dump_ast()` 的 `switch (n->type)` 里，`case N_EXPR:` 之后 | 新增 `case N_IF:` | 6 行 | `ast` 模式要能打印出 `如果` 树 |
| `exec()` 的 `switch (n->type)` 里，`case N_EXPR:` 之后 | 新增 `case N_IF:` | 7 行 | `run` 模式要能执行分支 |

合计 **25 行新增**，**0 行修改**（除了枚举尾部加逗号这一处字符级改动）。符合 INV-8："新代码只插入、不修改既有行"。

### 6.2 分段讲解

#### 第 1 段：给枚举加一个 `N_IF`

```diff
     N_LET,      /* let x = 1;        */
     N_ASSIGN,   /* x = 1;            */
-    N_EXPR      /* 表达式语句        */
+    N_EXPR,     /* 表达式语句        */
+    N_IF        /* if (条件) ... else ... */
 } NodeType;
```

**逐行解释**：

- `N_EXPR` 后面补了一个逗号——枚举的最后一个元素后面**不能**有逗号（虽然 C99 之后允许，但这里保持规范写法）。
- `N_IF` 放在**最后**，这样前面所有枚举值的数字都不变。虽然我们从来不依赖这些数字，但"只追加不改动"是好习惯。
- 注释里写清楚这个节点对应什么语法，方便后来的人（三个月后的你自己）。

**如果你写错了会怎样**：

- 忘了加逗号 → 编译报 `expected '}' before 'N_IF'`。
- 把 `N_IF` 插在中间 → 后面所有枚举值都变了。虽然不影响功能（没人拿数字当 key），但 diff 会变脏，破坏 INV-8。

#### 第 2 段：解析 `if`

```diff
+    /* if (条件) 语句 [else 语句] */
+    if (match(T_IF)) {
+        Node *n = node(N_IF, t);
+        expect(T_LP, "if 后需要 '('");
+        n->a = expression();
+        expect(T_RP, "if 条件后需要 ')'");
+        n->b = statement();          /* 条件为真时执行的分支 */
+        if (match(T_ELSE)) n->c = statement();  /* 条件为假时执行的分支（可以没有） */
+        return n;
+    }
```

**逐行解释**：

- `Token t = *cur();` 是 `statement()` 开头就有的（第 5 步代码），此时 `cur()` 指向 `if` 关键字。`node(N_IF, t)` 把这个 token 存进节点的 `tok` 字段——**出错时的行列号就靠它**。
- `match(T_IF)` 是第 2 步就定型的辅助函数：如果当前 token 是 `T_IF` 就 `pos++` 吃掉它并返回 1，否则什么都不做返回 0。**它不会报错**，所以可以用来试探。
- `expect(T_LP, "if 后需要 '('")`：当前 token 必须是 `(`，否则报错退出，并给出 `if 后需要 '('` 这样的说明。`expect` 也是第 2 步定型的。
- `n->a = expression();`：条件是**表达式**，所以走 `expression()`（第 4 步那套优先级链 `logical_or → logical_and → equal → compare → add → mul → unary → primary`）。这意味着 `if (x > 5 && y < 3)` 这种复合条件开箱即用。
- `n->b = statement();`：then 分支是**语句**，走 `statement()`。这就是第 5.2 节说的"分支是一条语句"。
- `if (match(T_ELSE)) n->c = statement();`：**只有真的吃到 `else` 关键字才赋值**。没吃到 `else` 时 `n->c` 保持 `node()` 里 `memset` 出来的 0，也就是 `NULL`。

**如果你写错了会怎样**：

- 忘了 `expect(T_LP, ...)` → 用户写 `if x > 5 { ... }` 时，`expression()` 会把 `x` 当成条件表达式解析成功，然后 `expect(T_RP)` 会在 `>` 处报错，错误信息会让人摸不着头脑。加上 `expect(T_LP)` 后，错误直接指向"if 后需要 '('"，非常清楚。这就是 `tests/case_04_missing_paren.ml` 期望的输出：

  ```
  错误 [1:4]: 语法错误：if 后需要 '('，当前是 'x'
  ```

- 条件写成 `n->a = statement();` → `if (x > 5)` 会被当成一条语句解析，`(` 开头会报"此处需要数字、变量、函数调用或括号表达式"。
- then 分支写成 `n->b = block();` → `if (x) print(1);`（不带花括号）会报"代码块需要 '{'"。

#### 第 3 段：打印 `if` 节点

```diff
+        case N_IF:
+            printf("如果\n");
+            kids[m++] = n->a;   /* 条件 */
+            kids[m++] = n->b;   /* then 分支 */
+            if (n->c) kids[m++] = n->c;   /* else 分支（可能没有） */
+            break;
```

**逐行解释**：

- `kids[]` 是 `dump_ast` 内部的局部数组（最多 128 个），收集要打印的子节点。
- `if (n->c)` 这个判断很关键：没有 `else` 时不要往 `kids` 里塞 NULL，否则打印树里会多出一个空行，读者会以为有第四个子节点。
- `break;` 之后是公共的"打印连接线 + 递归"代码，`dump_ast` 对所有节点类型是共用的。

**如果你写错了会怎样**：

- 忘了 `if (n->c)` → `dump_ast(NULL, ...)` 会被调用。虽然 `dump_ast` 开头有 `if (!n) return;` 不会崩，但打印出来的树里会多一个光秃秃的 `├── ` 开头、后面什么都没有的行。
- 忘了 `kids[m++] = n->b;` → 打印出来的 `如果` 节点只有条件，没有分支，读者会以为分支丢了。

#### 第 4 段：执行 `if`

```diff
+        case N_IF: {
+            /* 条件只求值一次：0 是假，非 0 是真 */
+            Value v = eval(n->a, e);
+            if (v.is_string) fatal_at(n->tok, "if 的条件必须是数字或比较表达式");
+            return exec(v.value ? n->b : n->c, e);
+            /* 没有 else 时 n->c 是 NULL，exec(NULL) 直接返回"什么都没发生" */
+        }
```

**逐行解释**：

- `case N_IF: {` 后面加 `{ }` 是为了能声明局部变量 `v`（C 的 `switch` 里所有 case 共享一个块作用域，不加花括号会跨 case 污染）。
- `eval(n->a, e)`：**条件只求值一次**，结果存进 `v`。
- `if (v.is_string) fatal_at(...)`：第 6 步走不到（第 5.4 节解释过），但为第 9 步铺路。**注意这里用的是 `n->tok`（`if` 关键字的位置），而不是字符串自己的位置**——语义上"条件不能是字符串"是 `if` 语句本身的毛病。
- `return exec(v.value ? n->b : n->c, e);`：三目运算符。`v.value` 为 0 选 `n->c`，否则选 `n->b`。**这就是 `if` 的全部实现**。
- `exec(NULL)` 之所以安全，靠的是 `exec` 开头那句 `if (!n) return noflow();`（第 5 步就写了，当时是为别的目的，现在正好派上用场）。

**如果你写错了会怎样**：

- 写成 `exec(eval(n->a, e).value ? n->b : n->c, e);` → 语法没错，但条件会被求值**一次以上**（三目运算符的两边只有一边求值，所以其实还是一次……但如果你写成 `if (eval(n->a,e).value) exec(n->b); else exec(n->c);` 这样的展开，就容易漏掉"else 分支为 NULL"的情况）。**始终推荐用 `exec(NULL)` 这套。**
- 忘了 `return` → `exec` 会继续往下走到 `return f;`，而 `f` 是 `noflow()`。看起来"能用"，但如果 `if` 出现在 `return` 语句之前，语义就错了。

### 6.3 完整 diff

> **怎么用这段 diff**：推荐**照着敲**——每改一小段就编译一次，比一次性粘贴更容易记住。
> 如果想直接应用，把下面的内容存成 `step.patch`，然后在**上一步的目录**里执行：
>
> ```sh
> patch -p1 < ../006/step.patch
> ```
>
> 应用完和本目录的 `minilang.c` **逐字节一致**（可以 `diff` 验证）。

<details>
<summary>点开看完整 diff（真实 diff -u 输出）</summary>

```diff
--- a/minilang.c
+++ b/minilang.c
@@ -271,7 +271,8 @@
     N_BLOCK,    /* 代码块 { ... }    */
     N_LET,      /* let x = 1;        */
     N_ASSIGN,   /* x = 1;            */
-    N_EXPR      /* 表达式语句        */
+    N_EXPR,     /* 表达式语句        */
+    N_IF        /* if (条件) ... else ... */
 } NodeType;
 
 typedef struct Node Node;
@@ -451,6 +452,17 @@
 
     /* 裸代码块 { ... } */
     if (cur()->type == T_LB) return block();
+
+    /* if (条件) 语句 [else 语句] */
+    if (match(T_IF)) {
+        Node *n = node(N_IF, t);
+        expect(T_LP, "if 后需要 '('");
+        n->a = expression();
+        expect(T_RP, "if 条件后需要 ')'");
+        n->b = statement();          /* 条件为真时执行的分支 */
+        if (match(T_ELSE)) n->c = statement();  /* 条件为假时执行的分支（可以没有） */
+        return n;
+    }
 
     /* let 变量 = 表达式; */
     if (match(T_LET)) {
@@ -543,6 +555,12 @@
             printf("表达式语句\n");
             kids[m++] = n->a;
             break;
+        case N_IF:
+            printf("如果\n");
+            kids[m++] = n->a;   /* 条件 */
+            kids[m++] = n->b;   /* then 分支 */
+            if (n->c) kids[m++] = n->c;   /* else 分支（可能没有） */
+            break;
     }
 
     for (int i = 0; i < m; i++) {
@@ -790,6 +808,14 @@
         case N_EXPR:
             (void)eval(n->a, e);
             return f;
+
+        case N_IF: {
+            /* 条件只求值一次：0 是假，非 0 是真 */
+            Value v = eval(n->a, e);
+            if (v.is_string) fatal_at(n->tok, "if 的条件必须是数字或比较表达式");
+            return exec(v.value ? n->b : n->c, e);
+            /* 没有 else 时 n->c 是 NULL，exec(NULL) 直接返回"什么都没发生" */
+        }
 
         default:
             fatal_at(n->tok, "这里不是一条语句");
```

</details>

> 顺便确认一下：diff 里**没有**文件头的变化。这是刻意设计的——`006/minilang.c` 和 `005/minilang.c` 的头注释一字不差，所以 diff 里只剩下真正的功能改动。文件头那句"文件头注释在教程所有步骤中保持完全一致"说的就是这件事。

### 6.4 插入位置说明

如果你要自己在 `005/minilang.c` 上手改，四处代码分别放在：

1. **枚举**：找到 `typedef enum {` 那一段，在 `N_EXPR      /* 表达式语句        */` 那一行下面加逗号并新增 `N_IF`。
2. **解析**：`static Node *statement(void)` 函数里，紧跟在 `if (cur()->type == T_LB) return block();` 和它后面的空行之后（也就是"裸代码块"那一段下面），插入 `if (match(T_IF)) { ... }` 整块。**不要**插到 `let` 分支后面——`if` 必须在 `let` 之前还是之后其实无所谓（它们靠 `match` 的 token 类型区分，互不干扰），但插在"裸代码块"后面读起来最顺。
3. **打印**：`static void dump_ast(Node *n, const char *prefix)` 的 `switch (n->type)` 里，在 `case N_EXPR:` 这个 case 的 `break;` 之后、`}` 之前。
4. **执行**：`static Flow exec(Node *n, Env *e)` 的 `switch (n->type)` 里，在 `case N_EXPR:` 这个 case 之后、`default:` 之前。

改完后编译：

```sh
cc -std=c11 -Wall -Wextra -O2 minilang.c -lm -o minilang
```

## 7. 测试

`./run.sh` 的真实输出：

```
▶ 用例（模式：run）
  ✅ case_01_if_else.ml           [run]
  ✅ case_02_if_only.ml           [run]
  ✅ case_03_nested.ml            [run]
  ✅ case_04_missing_paren.ml     [run]
▶ demo.ml
  | 1
✅ 全部通过（5 项）
```

逐个用例说明：

**`tests/case_01_if_else.ml`** —— 最基本的 if/else：

```
let x = 10;
if (x > 5) { print(1); } else { print(0); }
```

- 输入：`x = 10`，`x > 5` 求值为 `1`（真）。
- 期望 stdout：`1`（**注意末尾没有换行**——`print` 不自动换行，`tests/case_01_if_else.stdout` 文件里存的就是一个字符 `1`）。
- 期望 stderr：空。期望退出码：0。
- 为什么是这个结果：条件为真 → `exec(n->b)` → 执行 then 块 → `print(1)` 输出 `1`；else 分支的节点根本没被执行。

**`tests/case_02_if_only.ml`** —— 只有 if，没有 else：

```
let x = 1;
if (x > 5) { print(1); }
```

- 期望 stdout：**空文件**（一个字节都没有）。
- 期望退出码：0。
- 为什么是这个结果：`x > 5` 为假 → `exec(n->c)`，`n->c` 是 `NULL` → `exec` 立刻返回 `noflow()`。程序正常结束，什么都没打印。
- 这个用例专门用来验证"没有 else 时不会崩"。如果你把 `exec` 开头那句 `if (!n) return noflow();` 删掉，这个用例会段错误（`switch (n->type)` 解引用 NULL）。

**`tests/case_03_nested.ml`** —— 嵌套 if：

```
let x = 7;
if (x > 5) {
    if (x > 6) { print(2); } else { print(1); }
} else {
    print(0);
}
```

- 期望 stdout：`2`。期望退出码：0。
- 为什么：`x > 5` 为真 → 进 then 块 → 块里又是 `if (x > 6)`，`7 > 6` 为真 → 打印 `2`。最外层的 else 分支不会执行。

**`tests/case_04_missing_paren.ml`** —— 语法错误：

```
if x > 5 { print(1); }
```

- 期望 stderr（全文，一行）：

  ```
  错误 [1:4]: 语法错误：if 后需要 '('，当前是 'x'
  ```

- 期望 stdout：空。期望退出码：1。
- 为什么是 `[1:4]`：`if` 在第 1 行第 1 列，占 1~2 列；第 3 列是空格；`x` 在第 4 列。`expect(T_LP, ...)` 报的错用的是**当前 token** 的行列，也就是 `x` 的位置。
- 为什么会这样：`expect()` 的实现是"当前 token 不是我要的就 `fatal_at(*cur(), "语法错误：%s，当前是 '%s'", what, cur()->text)`"。`x` 是标识符，词法分析器用精确长度拷贝，所以 `text` 就是 `"x"`。

顺便说一个**你一定会觉得别扭的细节**：如果你把括号写全了但漏了右括号，比如 `if (x > 5 { print(1); }`，报错是：

```
错误 [1:11]: 语法错误：if 条件后需要 ')'，当前是 '{ '
```

注意 `'{ '` 后面**多了一个空格**！这是第 1 步词法分析器的一个小瑕疵：单字符运算符（`{` `}` `(` `)` `;` 等）走的是 `two[0] = c; two[1] = src[i+1];` 这条路，把**下一个字符**也塞进了 token 的 `text` 里。标识符和数字走的是另一条精确拷贝的路，所以 `x` 就干干净净。**记住这个，免得你以为自己的代码写对了。**

**`demo.ml` 冒烟测试** —— 输出 `1`（见第 3 节）。

## 8. 易错点与坑

### 坑 1：分支里忘了写 `{ }`，变量泄漏到外层

- **错误现象**：`if (x) let y = 1;` 之后 `print(y)` 本来应该报"未定义变量"，结果**打印出了值**。
- **原因**：`n->b = statement()` 拿到的是 `N_LET` 节点，`exec` 的 `case N_LET:` 调 `define_var(e, ...)`，而 `e` 就是外层环境。没有块就没有新作用域，`let` 直接落在外层。
- **正确写法**：永远写 `{ }`：

  ```
  if (x) { let y = 1; }
  ```

  这样 `y` 落在块自己的 `Env` 里（`case N_BLOCK:` 会 `env_init(&local, e)` 建一个新环境），块外就看不见了。

### 坑 2：`n->c` 没有 else 时是 NULL，别忘了判空

- **错误现象**：写 `n->c = statement();`（不加 `if (match(T_ELSE))`）→ 没有 `else` 时，`statement()` 会去解析下一个 token，把**后面的语句**当成 else 分支吃掉！比如：

  ```
  if (x > 5) { print(1); }
  print(2);
  ```

  会变成 `if` 的 else 分支是 `print(2);`，于是 `x > 5` 为假时也什么都不打印——`print(2)` 被吞掉了。

- **原因**：`match(T_ELSE)` 是个"试探"，不匹配时**什么都不做**；而 `statement()` 是"解析一条语句"，一定会消费 token。
- **正确写法**：`if (match(T_ELSE)) n->c = statement();`——只有确认有 `else` 才去解析分支。

### 坑 3：`exec` 里忘记 `return`，或者 `case` 忘了写花括号

- **错误现象**：在 `case N_IF:` 里声明 `Value v = ...` 但 `case` 后没写 `{` → 编译报错 `case label does not reduce to a switch expression` 或 `crosses initialization of 'Value v'`。
- **原因**：C 的 `switch` 里的所有 `case` 共享**同一个块作用域**，在 `case` 里直接声明变量会跨 case 冲突。
- **正确写法**：

  ```c
  case N_IF: {
      Value v = eval(n->a, e);
      ...
      return exec(v.value ? n->b : n->c, e);
  }
  ```

  注意 `case N_IF: {` 后面这个 `{` 和结尾的 `}` 都是必须的，缩进只是风格。

### 坑 4：给 `node()` 漏掉 `memset`，`n->c` 就不是 NULL

- **错误现象**：你在 `node()` 里"优化"掉了 `memset(n, 0, sizeof(*n))`，觉得反正每个字段都会赋值。结果 `n->c` 里是一段垃圾指针，`exec(n->c, e)` 直接段错误（而且是随机的、难以复现的那种）。
- **原因**：`node()` 分配的内存来自**静态数组** `nodes[]`，不是 `malloc`，内容是上一次运行残留的旧数据（甚至可能是 0，但不保证）。
- **正确写法**：保留 `memset`。这一行在第 2 步就写好了，注释里也说了"否则 switch 里没覆盖到的字段会是随机值，后面调试会非常痛苦"。

### 坑 5：`else` 写在了外层 `if` 后面，而你以为它属于内层

- **错误现象**：`if (a) if (b) print(1); else print(2);` 里，`b` 为假时什么都没打印。
- **原因**：悬垂 else——`else` 属于**最近**的 `if`。
- **正确写法**：想给外层配 else 就给内层加花括号：`if (a) { if (b) print(1); } else print(2);`

### 坑 6：以为 `if` 的条件可以是任意表达式

- **错误现象**：`if ("hi")` 报错，而且报错信息是"求值器还不支持字符串"而不是"条件必须是数字"。
- **原因**：`eval()` 在第 3 步就把字符串拦下来了（`case N_STR:` 直接 `fatal_at`），根本走不到 `exec` 里的 `is_string` 检查。
- **正确写法**：这不是你能改的（第 9 步之后字符串才有意义）。理解"错误在哪一层被拦下来"比记住报错文本更重要。

## 9. 自测题

### 9.1 概念题

**Q1.** `n->c` 在没有 `else` 时为什么是 `NULL`？这个 `NULL` 是谁写进去的？

**Q2.** 为什么 `exec()` 的第一行必须写 `if (!n) return noflow();`？如果删掉这一行，哪个测试用例会最先暴露问题？

**Q3.** `if (x > 5) let y = 1;` 和 `if (x > 5) { let y = 1; }` 在**运行时**有什么差别？为什么？

**Q4.** `if` 的条件里能不能写 `x > 5 && y < 3`？为什么能（或不能）？

<details>
<summary>参考答案</summary>

**A1.** `node()` 工厂函数里的 `memset(n, 0, sizeof(*n))` 把整个节点清零了。`n->c` 只有在 `match(T_ELSE)` 成功时才被显式赋值，否则保持 `memset` 留下的 0，也就是 `NULL`。

**A2.** 因为 `exec(v.value ? n->b : n->c, e)` 在没有 `else` 时会把 `NULL` 传进 `exec`。删掉这一行，`exec(NULL)` 会在 `switch (n->type)` 处解引用空指针 → 段错误。`tests/case_02_if_only.ml`（只有 if 没有 else）会最先暴露问题。

**A3.** 前者：`let y = 1;` 这条语句没有块，所以 `exec` 的 `case N_LET:` 直接在**外层**环境里定义 `y`，`y` 泄漏到外面。后来者：`{ ... }` 是 `N_BLOCK`，`exec` 会 `env_init(&local, e)` 建一个新环境，`y` 落在块自己的环境里，块外看不见（`print(y)` 会报"未定义变量 'y'"）。

**A4.** 能。条件走的是 `expression()`，它是第 4 步那套优先级链的最外层入口，包含了 `&&`（`logical_and`）和 `||`（`logical_or`）。`&&` 和 `||` 还带短路求值。

</details>

### 9.2 预测题（"改一行会发生什么"）

**Q5.** 把 `exec` 里那行 `return exec(v.value ? n->b : n->c, e);` 改成 `return exec(v.value ? n->b : n->b, e);`，会发生什么？

**Q6.** 把 `case N_IF` 里的 `Value v = eval(n->a, e);` 改成 `Value v = eval(n->a, e); v = eval(n->a, e);`（求值两次），`tests/case_01_if_else.ml` 的输出会变吗？

**Q7.** 把 `if (match(T_ELSE)) n->c = statement();` 改成 `n->c = statement();`，`tests/case_01_if_else.ml` 的输出会变吗？

**Q8.** 把 `n->b = statement();` 改成 `n->b = block();`，报什么错？

<details>
<summary>参考答案</summary>

**A5.** 条件为假时也会执行 then 分支。`if` 变成"不管条件如何都执行 then"。`tests/case_01_if_else.ml`（`x = 10`，条件为真）输出不变，仍是 `1`；但 `demo.ml` 同样不变。真正会变的是条件为假的场景，比如 `if (0) { print(1); } else { print(2); }` 会输出 `1` 而不是 `2`。可以自己写一个 `.ml` 验证。

**A6.** 输出不变（`1`）。因为 `tests/case_01_if_else.ml` 的条件 `x > 5` 没有副作用，求值两次结果一样。但如果你写 `if (print(9) || 1) { print(1); }`，输出会变成 `991`（`9` 被打两次）而不是 `91`。这就是"条件只求值一次"必须靠代码保证的原因。

**A7.** 会变。`match(T_ELSE)` 不匹配时 `pos` 不动，`statement()` 会去解析**下一条语句**当作 else 分支。`case_01_if_else.ml` 里 `if (x > 5) { print(1); } else { print(0); }` 后面没有别的语句，`statement()` 会去解析 `T_EOF`…… 实际结果是语法错误（`expression()` 遇到 EOF 报"此处需要数字、变量、函数调用或括号表达式"）。而在 `demo.ml` 那种 else 在最后的场景里也可能侥幸通过，但 `print(0);` 之后的语句会被吞掉。

**A8.** 报 `错误 [行:列]: 语法错误：代码块需要 '{'`，位置指向 then 分支的第一个 token。因为 `block()` 内部第一句是 `expect(T_LB, "代码块需要 '{'")`。

</details>

### 9.3 动手题（改代码 + 跑 tests 验证）

**Q9.** 在 `006/` 下新建一个 `my_cases/` 目录，写三个 `.ml` 文件并用 `./minilang` 跑出真实输出：

- `t1.ml`：只有 `if` 没有 `else`，条件为假 → 期望**没有输出**，退出码 0。
- `t2.ml`：`if (0) { print(1); } else { print(2); }` → 期望输出 `2`。
- `t3.ml`：`let x = 1; if (x) let y = 1; print(y);` → 期望输出 `1`（验证坑 1）。

**Q10.** 扩展 `dump_ast`：给 `如果` 节点的三条分支打上文字标签（比如 `条件` / `then` / `else`），让 `ast` 模式的输出更好读。改完后 `./minilang demo.ml ast` 的输出是什么样？

**Q11.**（进阶）给 MiniLang 加一个 `if` 的"块"形式支持：`if (x) { ... }` 里的 `let` 已经在块作用域了。请验证：**在 `if` 的分支里赋值**（`x = 5;` 而不是 `let x = 5;`）会改到外层的变量吗？写个 `.ml` 验证并解释为什么。

<details>
<summary>参考答案</summary>

**A9.** 三份 `.ml` 的真实运行结果：

```
$ ./minilang my_cases/t1.ml      # 无输出，退出码 0
$ ./minilang my_cases/t2.ml
2
$ ./minilang my_cases/t3.ml
1
```

**A10.** 一种做法是在 `dump_ast` 里给 `N_IF` 的子节点加前缀，比如把 `kids` 收集时改成先打印一行 `如果` 再分别用 `dump_ast` 打三棵子树，中间插 `printf("├─ 条件\n")` 之类的文字。改完之后输出大致是：

```
语法树：
代码块
├── 定义 x
│   └── 数字 10
└── 如果
    ├─ 条件
    │   └── 二元 GT
    │       ├── 变量 x
    │       └── 数字 5
    ├─ then
    │   └── 代码块
    │       └── 表达式语句
    │           └── 调用 print
    │               └── 数字 1
    └─ else
        └── 代码块
            └── 表达式语句
                └── 调用 print
                    └── 数字 0
```

（具体格式随你的实现而定，只要三种分支能区分开就算达标。）

**A11.** 会改到外层。`x = 5;` 解析出来是 `N_ASSIGN`，`exec` 的 `case N_ASSIGN:` 调 `set_var(e, ...)`，`set_var` 会**沿作用域链往外找**第一个叫 `x` 的变量并修改它。块里的 `let` 会新建变量（`define_var` 只在当前层），块里的赋值则会一路找到外层去改。

验证：

```
let x = 1;
if (1) { x = 5; }
print(x);      →  5
```

而如果块里写的是 `let x = 5;`，块外的 `x` 仍然是 `1`。

</details>

## 10. 本步的设计取舍

**为什么把 `if` 放在第 6 步，而不是和第 7 步的 `for` 一起？**

因为 `if` 是理解"语句 vs 表达式"的分水岭，而 `for` 的每一个部分（初始化、条件、更新）都是 `if` 已经讲过的东西。`for` 展开后是"一个 `let` + 一个 `while` + 一个 `if`"，先讲 `if` 再讲 `for`，读者只需要额外学"循环自带作用域"和"安全阀"两件事。

**为什么 `N_IF` 用 `a`/`b`/`c` 而不是给 `Node` 加专门的 `cond`/`then`/`else` 字段？**

加字段会让 `Node` 结构体变大（每个节点都多付 3 个指针 × 20000 个节点 = 480 KB），而且违背 INV-5"表结构一步定型"。用通用的 `a`/`b`/`c` 意味着**每种节点类型的语义由它自己解释**：`N_IF` 说"a 是条件"，`N_BINARY` 说"a 是左操作数"。代价是读代码时要查注释——但注释就在枚举里。

**代价：没有 `else if`。**

现在写 `else if` 得靠嵌套：

```
if (a) { ... } else { if (b) { ... } else { ... } }
```

多加一个 `N_ELSE` 节点类型会让 AST 变复杂，而且 C 里的 `else if` 本质就是嵌套 `if` 的语法糖。**我们选择让语法糖不存在**，这样 AST 树和源码结构一一对应，调试时不会出现"AST 里怎么多了一层"的困惑。

**还没做的：**

- 没有 `?:` 三目运算符（虽然 C 代码里用了三目，但那是实现细节，不是 MiniLang 的语法）。
- 没有 `if` 的块级 `else` 简写。
- 条件里不能放字符串（第 9 步之后 `if` 的条件检查会变成"必须是数字"，但那时 `eval` 也会返回字符串值，两个检查会同时生效）。
- 没有短路之外的条件求值优化（无所谓，解释器不在乎）。

## 11. 下一步预告

到这里，MiniLang 已经能"分支"了，但还只能从上到下顺序执行一遍。下一节我们加入 `for` 循环：`for (let i = 0; i < 10; i = i + 1) { ... }`，把 `for` 的三个部分分别对应到 AST 的三个字段，带你看懂 C 式 for 和等价 while 的展开关系，并且给循环加一个防止死循环的"安全阀"——写一个忘记写更新部分的 `for`，亲眼看看解释器是怎么在 100 万步后停下来报错的。

下一节：[第 7 步 · for 循环](../007/README.md)
