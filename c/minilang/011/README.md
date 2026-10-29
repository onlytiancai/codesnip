# 第 11 步 · return 与递归

> 进度 ██████████ 11/13 ｜ 上一节：[第 10 步 · 用户函数：定义与调用](../010/README.md) ｜ 下一节：[第 12 步 · 错误处理与诊断](../012/README.md)

## 1. 本步目标

读完你将能够：

- 写 `return 表达式;` 和 `return;` 两种返回语句，让函数**真正有返回值**。
- 理解 `Flow` 结构**从第 5 步就定型好了**——本步 `exec()` 的签名一个字都不用改。
- 写出递归函数：`factorial(5)` → `120`，`fib(10)` → `55`。
- 解释返回值是怎么**一层层冒泡**回调用点的。
- **亲眼看到**第 5 步"Env 用动态数组"这个决定的回报：
  递归 100 层稳稳当当，而固定数组版本第 8 层就段错误。

## 2. 运行方式

```sh
cd 011
./run.sh                 # 编译 + 跑全部用例 + 跑 demo
./minilang demo.ml       # 默认 run 模式
./minilang demo.ml ast   # 看递归调用的语法树
```

本步 `run.sh` 的真实输出见 [第 7 节](#7-测试)。

**本步还有一个对照实验**：

```sh
cd 011/experiments/fixed_env && ./run.sh
```

它会当场演示"如果作用域用固定数组会怎样"。

## 3. 现在的程序能做什么

第 10 步的函数只能打印，不能返回。这是最关键的一块拼图——**没有返回值，
函数就没法参与计算**（`factorial` 写不出来）。

| 能力 | 例子 | 状态 |
|---|---|---|
| 函数定义与调用 | `fn f(a) { ... }` | 第 10 步 |
| `return 表达式;` | `return a + b;` | **本步新增** |
| `return;`（无值） | `return;` | **本步新增** |
| 递归 | `factorial(5)` → 120 | **本步新增** |
| 函数里的 `return` | 提前结束函数 | **本步新增** |

`011/demo.ml` 的内容：

```
// 第 11 步：return 和递归
fn factorial(n) {
    if (n <= 1) {
        return 1;
    }
    return n * factorial(n - 1);
}
print(factorial(5));
```

真实运行：

```sh
$ ./minilang demo.ml
120
```

## 4. 前置知识回顾

第 5 步我们定型的 `Flow`：

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

当时这两行代码**一次也没被真正用上**——`returned` 永远是 0。
现在它们要上线了。

再看第 10 步调用函数的那段：

```c
call_depth++;
Flow flow = exec(d->a, &local);
call_depth--;
env_free(&local);

/* 还没有 return，所以函数的值暂时是 0（下一步会补上） */
return flow.returned ? flow.value : number(0);
```

**这行三元表达式就是本步的"接口"**：
`exec` 执行完函数体，把结果放在 `flow` 里；调用点看 `flow.returned` 决定用不用它。
本步要做的，只是让 `exec` **有可能**把 `flow.returned` 置成 1。

## 5. 本步引入的核心概念

### 5.1 概念一：为什么 `exec` 的签名不用改

如果第 5 步的 `exec` 返回的是 `void`，那本步就得改成：

```c
static Flow exec(Node *n, Env *e);   /* 签名变了 */
```

而**所有调用点都要跟着改**。数一下 `exec` 在代码里被调用了几次：
`exec` 的 `N_BLOCK` 分支里一次、`N_IF` 分支一次、`N_FOR` 分支里两次、
`eval` 的 `N_CALL` 分支里一次、`mode_run` 里一次——**7 处**。

那会是一大堆"为了支持 return 而不得不改的签名和调用点"，
把 diff 搅得面目全非，读者根本看不出"本步到底加了什么功能"。

因为第 5 步提前把 `Flow` 定型好了，本步的改动只有：

```diff
--- a/minilang.c
+++ b/minilang.c
@@ -287,7 +287,8 @@
     N_EXPR,     /* 表达式语句        */
     N_IF,       /* if (条件) ... else ... */
     N_FOR,      /* for (init; cond; update) { ... } */
-    N_FN        /* fn 名字(参数) { ... } */
+    N_FN,       /* fn 名字(参数) { ... } */
+    N_RETURN    /* return 表达式; */
 } NodeType;
 
 typedef struct Node Node;
@@ -468,6 +469,14 @@
     /* 裸代码块 { ... } */
     if (cur()->type == T_LB) return block();
 
+    /* return 表达式; */
+    if (match(T_RETURN)) {
+        Node *n = node(N_RETURN, t);
+        if (cur()->type != T_SEMI) n->a = expression();   /* return; 也可以不带值 */
+        expect(T_SEMI, "return 语句末尾需要 ';'");
+        return n;
+    }
+
     /* 函数定义：fn 名字(参数, ...) { ... } */
     if (match(T_FN)) {
         Node *n = node(N_FN, t);
@@ -655,6 +664,10 @@
             printf(")\n");
             kids[m++] = n->a;   /* 函数体 */
             break;
+        case N_RETURN:
+            printf("返回\n");
+            if (n->a) kids[m++] = n->a;   /* 返回的表达式（可以没有） */
+            break;
     }
 
     for (int i = 0; i < m; i++) {
@@ -1080,6 +1093,14 @@
             /* 函数定义在执行前就登记好了，这里什么都不用做 */
             return f;
 
+        case N_RETURN:
+            /* return 只能出现在函数里 */
+            if (call_depth == 0)
+                fatal_at(n->tok, "return 只能在函数内部使用");
+            f.returned = 1;
+            f.value = n->a ? eval(n->a, e) : number(0);
+            return f;   /* 这个 f 一路往上冒泡，直到函数调用处被接住 */
+
         case N_FOR: {
             /* 循环自带一个作用域：i 只在循环里活着 */
             Env local;
```

**整个本步只增加了 30 行左右，而且没有一行是"改别人的"**——
这就是 INV-3 的价值。

### 5.2 概念二：返回值怎么一层层冒泡

`return` 的值不是通过 C 的返回值机制传出去的（`exec` 返回的是 `Flow`），
而是**沿着 `exec` 的调用链往上传**。

以 `factorial(5)` 为例：

```text
exec(函数体)                      ← 里面执行到 return n * factorial(n-1)
   │
   ├─ eval(n * factorial(n-1))
   │     │
   │     └─ eval(调用 factorial(4))  ← 递归，又是一个 exec
   │           │
   │           └─ exec(函数体)
   │                 └─ 遇到 return 1  →  f.returned = 1, f.value = 1
   │                       │
   │                       └─ 返回 f（returned=1, value=1）给调用点
   │
   ├─ eval 拿到 1，算出 5 * 1 = 5
   │
   └─ f.returned = 1; f.value = 5;  →  继续往上冒泡
```

**每一层 `exec` 都要检查子调用有没有 `returned`**。第 5 步的代码块分支已经写好了：

```c
case N_BLOCK: {
    Env local;
    env_init(&local, e);
    for (int i = 0; i < n->count; i++) {
        f = exec(n->items[i], &local);
        if (f.returned) { env_free(&local); return f; }   /* ← 接住并往上传 */
    }
    env_free(&local);
    return noflow();
}
```

**如果没有 `if (f.returned)` 这句**，函数体会继续执行后面的语句，
`return` 就形同虚设了。

### 5.3 概念三：三个地方都要"接住" return

`return` 可能出现在函数体的**任何一层嵌套**里，所以每一层都要检查：

| 位置 | 代码 | 本步之前 |
|---|---|---|
| 代码块 | `if (f.returned) { env_free(&local); return f; }` | 第 5 步就有 ✔ |
| `if` 分支 | `return exec(v.value ? n->b : n->c, e);` | 第 6 步就有 ✔ |
| `for` 循环 | `if (f.returned) { env_free(&local); return f; }` | 第 7 步就有 ✔ |
| 函数调用 | `flow.returned ? flow.value : number(0)` | 第 10 步就有 ✔ |

**这四处都是"提前定型的"**——写它们的时候，`return` 还不存在，
但每处都留了"如果子执行有返回，就立刻往上传"的口子。

**这就是"提前设计"的红利**：本步一行都不用改这四处。

### 5.4 概念四：`return;` 和 `return 表达式;`

```c
/* return 表达式; */
if (match(T_RETURN)) {
    Node *n = node(N_RETURN, t);
    if (cur()->type != T_SEMI) n->a = expression();   /* return; 也可以不带值 */
    expect(T_SEMI, "return 语句末尾需要 ';'");
    return n;
}
```

**逐行解释**：

- `if (cur()->type != T_SEMI)` —— 如果 `return` 后面**不是**分号，
  就说明还有一个表达式要算。没有表达式时 `n->a` 保持 `NULL`。
- `n->a ? eval(n->a, e) : number(0)` —— 执行时，有表达式就求值，
  没有就返回 0。

**注意 `n->a` 为 NULL 是靠 `memset` 保证的**（第 5 步 `node()` 里的
`memset(n, 0, sizeof(*n))`）。没写 `n->a = NULL;` 但它确实是 NULL。

`return;` 的效果是"提前结束函数，返回 0"：

```
fn nothing() { return; }
print(nothing());        →  0
```

### 5.5 概念五：`return` 不能出现在函数外面

```c
case N_RETURN:
    /* return 只能出现在函数里 */
    if (call_depth == 0)
        fatal_at(n->tok, "return 只能在函数内部使用");
```

**怎么知道"当前不在函数里"？** 看 `call_depth`：

- 第 10 步的 `call_depth` 在**调用函数时**加一、返回时减一；
- 所以 `call_depth == 0` 就意味着" we're 处在最外层"，也就是函数外面。

真实输出：

```
错误 [1:1]: return 只能在函数内部使用
```

退出码 `1`。

### 5.6 概念六：递归

递归 = 函数调用自己。三要素：

1. **终止条件**：`if (n <= 1) { return 1; }`
2. **递归调用**：`factorial(n - 1)`
3. **用递归结果继续计算**：`n * factorial(n - 1)`

```
fn factorial(n) {
    if (n <= 1) { return 1; }       /* 终止条件 */
    return n * factorial(n - 1);     /* 递归 + 用结果 */
}
```

调用链：

```text
factorial(5)
  └─ 5 * factorial(4)
       └─ 4 * factorial(3)
            └─ 3 * factorial(2)
                 └─ 2 * factorial(1)
                      └─ return 1        ← 终止，开始往回算
                 ← 2 * 1 = 2
            ← 3 * 2 = 6
       ← 4 * 6 = 24
  ← 5 * 24 = 120
```

**每一层递归都有自己的作用域**：`factorial(4)` 的 `n` 和 `factorial(5)` 的 `n`
是两个不同的变量，在两个不同的 `Env` 里。

### 5.7 概念七：第 5 步那个决定的回报

现在来看第 5 步埋下的伏笔。

第 5 步我们把 `Env` 的变量表做成了**动态数组**（`Var *vars`），
而不是固定数组（`Var vars[2048]`）。当时的理由只是"省内存"，
真正的理由现在才显出来：

| | 固定数组 `Var vars[2048]` | 动态数组 `Var *vars` |
|---|---|---|
| `sizeof(Env)` | ≈ **272 KB** | **24 字节** |
| 每递归一层压栈 | 2 个 Env ≈ **545 KB** | 2 个 Env ≈ **48 字节** |
| 8 MB 默认栈能撑几层 | **约 8~15 层** | 几百层没问题 |
| `MAX_CALL_DEPTH = 128` | **够不着**（先段错误） | 真正生效 |

也就是说，如果第 5 步选了固定数组，那么：

- `factorial(5)` 还能跑（只有 6 层）；
- `factorial(20)` 就可能段错误；
- `MAX_CALL_DEPTH = 128` 这个保护**根本用不上**，因为程序在到达 128 层之前
  就已经被操作系统干掉了（SIGSEGV，退出码 139）。

**这个对照实验在 `011/experiments/fixed_env/` 里**，跑一下就能看到：

```
递归深度 5     固定数组：正常，输出 5             动态数组：正常，输出 5
递归深度 8     固定数组：💥 段错误（exit 139，栈溢出）  动态数组：正常，输出 8
递归深度 10    固定数组：💥 段错误（exit 139，栈溢出）  动态数组：正常，输出 10
递归深度 20    固定数组：💥 段错误（exit 139，栈溢出）  动态数组：正常，输出 20
递归深度 100   固定数组：💥 段错误（exit 139，栈溢出）  动态数组：正常，输出 100
```

自己跑一遍：

```sh
cd 011/experiments/fixed_env
./run.sh
```

### 5.8 概念八：`call_depth` 的配对

第 10 步加了 `call_depth++` / `call_depth--`。本步加了 `return`，
**必须重新检查一遍所有提前返回的路径**。

看 `eval` 的 `N_CALL` 分支：

```c
call_depth++;
Flow flow = exec(d->a, &local);
call_depth--;
env_free(&local);
```

这里没问题——`exec` 无论怎么返回（正常返回或 `return` 冒泡），
都会回到这一行，`call_depth--` 一定会执行。

**但如果有这样写就会漏**：

```c
call_depth++;
if (something_wrong) return number(0);   /* ← 这里漏了 call_depth-- */
Flow flow = exec(d->a, &local);
call_depth--;
```

本步没有引入这种路径，所以不用改。**但这是加 `return` 时必须检查的清单**：
凡是能在函数调用中间提前返回的地方，都要确认计数器配对。

## 6. 动手：把代码加进来

### 6.1 改动清单

| 位置 | 改了什么 | 行数 |
|---|---|---|
| `NodeType` 枚举 | 加 `N_RETURN` | +2 |
| `statement()` | 加 `return` 分支 | +8 |
| `dump_ast()` | 加 `N_RETURN` 的 `case` | +5 |
| `exec()` | 加 `case N_RETURN:` | +9 |

合计约 **+25 行**。

**本步没有一行是"修改既有代码"**——这是整套教程里最漂亮的一步。
第 5 步定型的 `Flow` 让这一切成为可能。

### 6.2 分段讲解

#### 6.2.1 `return` 的语法分析

```c
/* return 表达式; */
if (match(T_RETURN)) {
    Node *n = node(N_RETURN, t);
    if (cur()->type != T_SEMI) n->a = expression();   /* return; 也可以不带值 */
    expect(T_SEMI, "return 语句末尾需要 ';'");
    return n;
}
```

**逐行解释**：

- `if (cur()->type != T_SEMI)` —— 这是判断"有没有表达式"的关键。
  `return;` 后面紧跟分号，`return 1;` 后面是表达式。
- **放置位置**：在 `statement()` 里，`fn` 分支之前。
  顺序无所谓（每个分支看的是不同的 token），但放在一起好找。

**如果你写错了会怎样**：写成 `if (cur()->type == T_SEMI) n->a = expression();`
（把 `!=` 写成 `==`），那么 `return 1;` 里的 `1` 会被当成"没有表达式"，
`expect(T_SEMI)` 立刻报"语法错误：return 语句末尾需要 ';'，当前是 '1'"。

#### 6.2.2 `exec` 里的 `N_RETURN`

```c
case N_RETURN:
    /* return 只能出现在函数里 */
    if (call_depth == 0)
        fatal_at(n->tok, "return 只能在函数内部使用");
    f.returned = 1;
    f.value = n->a ? eval(n->a, e) : number(0);
    return f;   /* 这个 f 一路往上冒泡，直到函数调用处被接住 */
```

**逐行解释**：

- `f` 是函数开头就准备好的 `Flow`（`Flow f = noflow();`）。
  这里把它改成"有返回值"的状态。
- `f.value = n->a ? eval(n->a, e) : number(0);` ——
  **注意求值顺序**：`n->a` 先被判断（是不是有表达式），
  有才调 `eval`。C 的 `?:` 是短路的，所以 `eval` 不会对 `NULL` 调用。
  （`eval` 开头有 `if (!n) return number(0);` 兜底，所以即使调了也安全，
  但短路更清楚。）
- `return f;` —— 把这个 `Flow` 返回给调用者。
  调用者（第 5 步的代码块、第 7 步的循环、第 10 步的函数调用）会检查 `returned`。

**如果你写错了会怎样**：忘了 `f.returned = 1;`，
那么值虽然算出来了，但上层看到 `returned == 0` 会当成"什么都没发生"，
函数体会继续往下执行，`return` 形同虚设。

#### 6.2.3 `dump_ast` 的新分支

```c
case N_RETURN:
    printf("返回\n");
    if (n->a) kids[m++] = n->a;   /* 返回的表达式（可以没有） */
    break;
```

**逐行解释**：`if (n->a)` 是必须的——`return;` 没有表达式，
如果无条件 `kids[m++] = n->a;`，就会把一个 `NULL` 塞进 `kids`，
`dump_ast` 虽然有 `if (!n) return;` 不会崩，但会打印出一个空节点，
让读者困惑。

### 6.3 完整 diff

> **怎么用这段 diff**：推荐**照着敲**——每改一小段就编译一次，比一次性粘贴更容易记住。
> 如果想直接应用，把下面的内容存成 `step.patch`，然后在**上一步的目录**里执行：
>
> ```sh
> patch -p1 < ../011/step.patch
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
| `N_RETURN` | `NodeType` 枚举末尾 |
| `return` 分支 | `statement()` 里，`fn` 分支之前 |
| `N_RETURN` 的 `case` | `dump_ast()` 的 `switch` 末尾 |
| `case N_RETURN:` | `exec()` 的 `switch` 里，`case N_FN:` 之后 |

## 7. 测试

`./run.sh` 的真实输出：

```
▶ 用例（模式：run）
  ✅ case_01_factorial.ml         [run]
  ✅ case_02_fibonacci.ml         [run]
  ✅ case_03_deep_recursion.ml    [run]
  ✅ case_04_return_outside.ml    [run]
  ✅ case_05_recursion_limit.ml   [run]
  ✅ case_06_bare_return.ml       [run]
▶ demo.ml
  | 120
✅ 全部通过（7 项）
```

逐个用例说明：

### case_01：阶乘

```
fn factorial(n) {
    if (n <= 1) { return 1; }
    return n * factorial(n - 1);
}
print(factorial(5));
```

输出 `120`。

### case_02：斐波那契

```
fn fib(n) {
    if (n < 2) { return n; }
    return fib(n - 1) + fib(n - 2);
}
print(fib(10));
```

输出 `55`。注意这个函数**一次调用会展开成指数级数量的调用**
（`fib(10)` 大概要展开上千次），但每次调用的栈深度只有 10 层。

### case_03：深度 100 的递归

```
fn depth(n) {
    if (n <= 0) { return 0; }
    return 1 + depth(n - 1);
}
print(depth(100));
```

输出 `100`。**这个用例是第 5 步那个动态数组决定的直接证明**——
换成固定数组版本它在第 8 层就段错误了（见 `011/experiments/`）。

### case_04：函数外面用 return（错误用例）

```
return 1;
```

期望 stderr：

```
错误 [1:1]: return 只能在函数内部使用
```

期望退出码 `1`。

### case_05：递归深度超限（错误用例）

```
fn forever(n) { return forever(n + 1); }
forever(0);
```

期望 stderr：

```
错误 [1:24]: 函数递归深度超过 128
```

期望退出码 `1`。注意位置是**第 1 行第 24 列**——那是 `forever` 这个调用
所在的位置（`fn forever(n) { return forever(n + 1); }` 里 `forever` 的起始列）。

### case_06：不带值的 return

```
fn nothing() { return; }
print(nothing());
```

输出 `0`。

### 自己动手试

```sh
cd 011
./minilang demo.ml ast
```

```
语法树：
代码块
├── 函数 factorial(n)
│   └── 代码块
│       ├── 如果
│       │   ├── 二元 LE
│       │   │   ├── 变量 n
│       │   │   └── 数字 1
│       │   └── 代码块
│       │       └── 返回
│       │           └── 数字 1
│       └── 返回
│           └── 二元 STAR
│               ├── 变量 n
│               └── 调用 factorial
│                   └── 二元 MINUS
│                       ├── 变量 n
│                       └── 数字 1
└── 表达式语句
    └── 调用 print
        └── 调用 factorial
            └── 数字 5
```

看这棵树就能明白递归在语法上一点都不特殊——
`factorial` 的函数体里就是一次普通的函数调用。

## 8. 易错点与坑

### 坑 1：`return` 之后代码还在执行

**现象**：

```
fn f() { return 1; print(999); }    ← 999 不会被打印（MiniLang 会报错）
```

**原因**：在 MiniLang 里 `return` 之后的语句**仍然会被解析**，
所以 `print(999)` 会被当成一条语句，只是执行时到不了那里。

**正确写法**：没有"不可达代码"的警告机制，但你可以自己注意：
`return` 应该是函数的最后一条语句。

### 坑 2：忘了 `f.returned = 1`

**现象**：`return` 里的值算出来了，但函数返回 0。

**原因**：上层（代码块 / 循环 / 调用点）靠 `returned` 标志判断
"要不要用这个值"，标志没置就当成"什么都没发生"。

**正确写法**：`f.returned = 1;` 和 `f.value = ...` 要**成对出现**。

### 坑 3：`call_depth--` 被跳过

**现象**：调用几次之后莫名报"递归深度超过 128"。

**原因**：某条提前返回的路径漏了 `call_depth--`。

**正确写法**：加 `return` 之后，把 `eval` 的 `N_CALL` 分支重新审一遍，
确认 `call_depth++` 之后的所有出口都配对减一。

### 坑 4：以为 `return` 能返回字符串

**现象**：`return "hi";` 报"变量目前只支持数字类型"。

**原因**：`Value` 虽然有 `is_string` 字段，但第 9 步只让字符串出现在 `print` 里。
`exec` 的 `N_RETURN` 分支直接把 `eval` 的结果塞进 `f.value`——
如果那是字符串值，调用点会把它当数字用。

**正确写法**：教学版里 `return` 只能返回数字。要返回字符串，
得先让 `Value` 的字符串真正成为一等类型（第 9 步的限制）。

### 坑 5：递归写成没有终止条件

**现象**：报"函数递归深度超过 128"，而不是你以为的"无限递归"。

**原因**：`call_depth` 计数器挡住了它。

**正确写法**：写递归前先想清楚"什么情况下不再递归"。

### 坑 6：在 `for` 循环里 `return`

**现象**：`return` 之后循环后面的语句不执行了。

**原因**：这是**正确行为**。`return` 应该结束整个函数，不只是跳出循环。

**正确写法**：如果你想"跳出循环但继续执行函数"，MiniLang 没有 `break`/`continue`
——这是语言的一个缺口，可以自己加一个 `N_BREAK` 节点试试。

## 9. 自测题

### 9.1 概念题

**Q1.** 为什么本步 `exec()` 的签名一个字都没改？

<details><summary>参考答案</summary>

因为第 5 步就把 `Flow { int returned; Value value; }` 定型好了，
`exec()` 从第一天起就返回 `Flow`。
本步只是让 `returned` 有机会变成 1，不需要改签名，也不需要改 7 个调用点。

</details>

**Q2.** `return` 的值是怎么传回调用点的？

<details><summary>参考答案</summary>

不是靠 C 的返回值，而是靠 `Flow` 结构**沿着 `exec` 的调用链往上冒泡**：
`case N_RETURN` 把值放进 `f.value` 并返回 `f`；
上层的代码块 / `if` / `for` 检查 `f.returned`，是 1 就立刻 `return f` 继续往上传；
最后到函数调用点，被 `flow.returned ? flow.value : number(0)` 接住。

</details>

**Q3.** `return;` 和 `return 0;` 有什么区别？

<details><summary>参考答案</summary>

在 MiniLang 里**没有区别**——`return;` 的 `n->a` 是 `NULL`，
执行时 `f.value = n->a ? eval(n->a, e) : number(0);` 会得到 `number(0)`。
所以 `return;` 等价于 `return 0;`。
在 C 里它们完全不同（`void` 函数 vs 返回 0 的函数）。

</details>

### 9.2 预测题（改一行会发生什么）

**Q4.** 把 `case N_RETURN` 里的 `f.returned = 1;` 删掉，`case_01` 会输出什么？

<details><summary>参考答案</summary>

输出 `0`。
值算出来了（`f.value = 120`），但 `returned` 还是 0，
所以代码块不会往上传，函数调用点看到 `flow.returned == 0`，
返回 `number(0)`。

</details>

**Q5.** 把 `if (call_depth == 0)` 这个检查删掉，顶层的 `return 1;` 会怎样？

<details><summary>参考答案</summary>

不会报错，而是**静默地**结束当前这条顶层语句。
`mode_run` 里有一句 `if (f.returned) fatal_at(..., "return 不能出现在函数外部")`，
所以还是会被抓住——但报错信息变成"return 不能出现在函数外部"，
而不是更准确的"return 只能在函数内部使用"。

</details>

**Q6.** 把 `f.value = n->a ? eval(n->a, e) : number(0);` 改成 `f.value = number(0);` 会怎样？

<details><summary>参考答案</summary>

`return` 永远返回 0，`case_01` 输出 `0` 而不是 `120`。
这题是在考你能不能一眼看出 `?:` 两边的作用。

</details>

### 9.3 动手题

**Q7.** 写一个 `fn sum_to(n)`，返回 1 + 2 + ... + n：

```
fn sum_to(n) {
    if (n <= 0) { return 0; }
    return n + sum_to(n - 1);
}
print(sum_to(10));
```

预期输出 `55`。

<details><summary>参考答案</summary>

输出 `55`。

</details>

**Q8.** 写一个 `fn is_even(n)`，用递归判断奇偶（不能取模）：

```
fn is_even(n) {
    if (n == 0) { return 1; }
    return is_even(n - 2);
}
print(is_even(10));
```

预期输出 `1`（1 表示偶数）。

<details><summary>参考答案</summary>

输出 `1`。
注意 `is_even(1)` 会一直递归到 `n == 0` 永远不成立，
最终撞上"递归深度超过 128"——想处理奇数就要加一个 `n == 1` 的分支。

</details>

**Q9.** 跑 `011/experiments/fixed_env/run.sh`，把输出贴进你的笔记。然后回答：
为什么"固定数组版"在深度 8 就崩，而"动态数组版"跑到 100 都没事？

<details><summary>参考答案</summary>

因为 `Env` 是 C 栈上的局部变量。
固定数组版 `sizeof(Env)` ≈ 272 KB，每递归一层压两个（调用处一个 + 函数体的代码块一个），
约 545 KB/层，8 MB 的栈只能撑 8~15 层；
动态数组版 `sizeof(Env)` 只有 24 字节，栈上几乎不占地方。

</details>

### 9.4 延伸题

**Q10.** 想给 MiniLang 加 `break` 和 `continue`（跳出循环 / 跳过本次迭代），
你打算怎么设计？提示：想想 `Flow` 里除了 `returned` 还可以加什么。

<details><summary>参考答案</summary>

可以在 `Flow` 里加两个标志，比如 `int broke; int continued;`，
再加两个 AST 节点 `N_BREAK` / `N_CONTINUE`。
`exec` 的 `N_BREAK` 分支置 `f.broke = 1`，
`N_FOR` 分支在循环体执行后检查 `f.broke` 决定要不要跳出。
难点在于：`break` 出现在 `if` 里面时，信息也要一层层冒泡，
所以 `Flow` 的三个标志要一起冒泡。

</details>

## 10. 本步的设计取舍

- **`return;` 和 `return 0;` 等价**：C 里它们不同，MiniLang 里相同。
  这是"简化语言"的代价，README 里必须写清楚，否则读者会踩坑。
- **只支持顶层函数**：递归只能靠顶层函数实现，嵌套函数还不支持。
- **`MAX_CALL_DEPTH = 128`**：这个数字是"够用又不至于把栈撑爆"的经验值。
  真正能跑多少层取决于第 5 步的 `Env` 大小——**语言的限制和环境是耦合的**。
- **`call_depth` 而不是栈深度检查**：用计数器比直接测栈更可控，
  但它数的是"函数调用层数"，不是"实际栈深度"。教学版够用了。

## 11. 下一步预告

现在 MiniLang 的**语言能力已经完整**了：变量、作用域、控制流、函数、递归、
字符串模板、内置函数，一个能写实际程序的解释器。

但它的错误提示还比较糙：

```
错误 [1:9]: 未定义变量 'price'
```

用户还得自己数第 9 列是哪一列。

下一节：[第 12 步 · 错误处理与诊断](../012/README.md)，我们会加上：

- 出错时**把出错的那一行源码和 `^` 指示符**一起打出来；
- 把整套教程的错误**分类整理**成一张表；
- 讲清楚 `error_at`（打印后继续）和 `fatal_at`（打印后退出）的区别；
- 讨论"遇到第一个错误就停"这个策略的取舍。
