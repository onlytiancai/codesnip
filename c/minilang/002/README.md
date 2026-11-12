# 第 2 步 · 表达式语法分析 → AST

> 进度 ██░░░░░░░░ 2/13 ｜ 上一节：[第 1 步 · 词法分析器](../001/README.md) ｜ 下一节：[第 3 步 · AST 求值器（算术）](../003/README.md)

## 1. 本步目标

读完你将能够：

- 让 `./minilang demo.ml ast` 把一个 MiniLang 表达式打印成一棵**抽象语法树（AST, Abstract Syntax Tree）**，
  带 `├──` / `└──` 连接线。
- 说清楚 `1 + 2 * 3` 会被解析成什么形状的树，以及为什么是这个形状（运算符优先级）。
- 用**递归下降（recursive descent）**写出一个能处理全部 6 层二元运算符优先级的解析器。
- 用**节点池（node pool）**分配 AST 节点，并解释为什么 `memset` 清零这一步不能省。
- 区分 `match()`（试探性匹配）和 `expect()`（必须匹配，否则报错）两个不同的取 token 方式。
- 遇到语法错误时给出**带行列号**的错误信息，并以退出码 `1` 结束。

## 2. 运行方式

```sh
cd 002
./run.sh                 # 编译 + 跑全部用例 + 跑 demo
./minilang demo.ml      # 默认 run 模式（本步还没实现）
./minilang demo.ml tokens
./minilang demo.ml ast
```

## 3. 现在的程序能做什么

| 模式 | 输入 | 输出 | 退出码 |
|---|---|---|---|
| `tokens` | 一个 `.ml` 文件 | token 列表（和第 1 步一样，词法器**一行没改**） | 0 / 1 |
| `ast` | **只有一个表达式**的 `.ml` 文件 | stdout：语法树；出错时 stderr：`错误 [行:列]: 语法错误：…` | 0 / 1 |
| `run`（默认） | 一个 `.ml` 文件 | stderr `解释执行尚未实现，请先用 tokens 或 ast 模式查看结果` | 1 |

真实运行：

```console
$ printf '1 + 2 * 3\n' > /tmp/e.ml
$ ./minilang /tmp/e.ml ast
语法树：
二元 PLUS
├── 数字 1
└── 二元 STAR
    ├── 数字 2
    └── 数字 3
```

注意 `ast` 模式**只接受一个表达式**——它不处理语句、不处理分号、不处理 `let`。
第 5 步会把 `ast` 模式扩成能解析整个程序。

## 4. 前置知识回顾

上一步我们已经把源码切成了 token 流，`Token` 结构体里带着 `type`、`text`、`num`、
以及**行列号** `line` / `col`。这一步要做的是：

- 用一个全局游标 `pos` 指向"当前读到第几个 token"（第 1 步的 `ntok` 是"一共多少个"，
  方向相反，别搞混）。
- 定义一个**节点类型 `NodeType` 枚举**和**节点结构体 `Node`**，它俩和 `Token` 是两套东西：
  `Token` 是**词法**的单位，`Node` 是**语法**的单位。

`Token` 的行列号在这一步终于派上用场了：语法错误发生时，我们报的是
**当前 token 的行列号**，也就是"出错那个词在源码里的位置"。

## 5. 本步引入的核心概念

### 5.1 为什么要 AST，而不是"边解析边执行"？

一个很自然的想法是：**为什么要先造一棵树？直接递归解析、解析到哪算到哪不行吗？**

比如你想直接执行 `1 + 2 * 3`，可以这样写：

```text
解析器读到 PLUS → 递归解析左边 → 递归解析右边 → 相加
```

问题在于，**你不知道该不该递归解析右边**——得先知道 `+` 的右边是不是还有
`*` 需要先算。如果不先把整棵树建出来，你就没法回头调整。

更要命的是**优先级**。`1 + 2 * 3` 到底是 `(1+2)*3`（=9）还是 `1+(2*3)`（=7）？
一个"边解析边执行"的程序必须**事先**知道答案，否则解析到 `+` 之后发现右边是 `*`，
已经来不及了。

而 AST 把这件事拆成了两步：

```text
   token 流                  AST                    求值
  ──────────         ────────────────────      ─────────────
  1 + 2 * 3      →    二元 PLUS               →    7
                       ├── 数字 1
                       └── 二元 STAR
                           ├── 数字 2
                           └── 数字 3
   （只管形状）        （只管形状）             （只管数值）
```

**分开的三个好处：**

| 好处 | 说明 | 哪一步享受到 |
|---|---|---|
| 优先级被"固化"成树形 | 求值器不需要知道优先级规则，它只要"从叶子往回算" | 第 3 步 |
| 可以打印出来给人看 | `ast` 模式就是靠这个——树是"可以展示的中间表示" | 第 2 步 |
| 将来可以换后端 | 同一棵 AST 可以解释执行，也可以编译成字节码 | 第 13 步的展望 |

第 3 步你会亲眼看到第一个好处：**求值器里一行优先级代码都没有**。
`+` 和 `*` 谁先算，完全由树形决定。

### 5.2 Node 结构体：每个字段的用途

```c
typedef struct Node Node;
struct Node {
    NodeType type;
    Token   tok;              /* 出错定位用：行列号 */
    double  num;              /* N_NUM 的数值      */
    char    text[MAX_TEXT];   /* N_VAR/N_CALL 的名字，N_STR 的内容 */
    int     op;               /* 运算符的 TokType  */
    Node   *a, *b, *c;        /* 子节点            */
    Node   *items[128];       /* N_CALL 的实参列表 */
    int     count;
    char   *params[MAX_PARAMS]; /* 函数参数名（后面几步才用） */
    int     nparams;
};
```

| 字段 | 谁用 | 说明 |
|---|---|---|
| `type` | 所有人 | 这是哪种节点（`N_NUM` / `N_VAR` / …） |
| `tok` | 错误处理 | **每个节点都记住自己是从哪个 token 来的**，出错时能报准确位置 |
| `num` | `N_NUM` | 数字字面量的值 |
| `text` | `N_VAR` / `N_CALL` / `N_STR` | 变量名 / 函数名 / 字符串内容 |
| `op` | `N_UNARY` / `N_BINARY` | 运算符的 `TokType`（`T_PLUS` / `T_STAR` …） |
| `a, b, c` | `N_UNARY` / `N_BINARY` | 子节点。`a` 是左/唯一操作数，`b` 是右操作数，`c` 备用 |
| `items[]` / `count` | `N_CALL` | 函数调用的实参列表 |
| `params[]` / `nparams` | 第 10 步 | 函数参数名——**本步还没用上，提前留好** |

**为什么用 `a, b, c` 而不是 `left, right`？**
因为不同节点需要的子节点数量不一样：一元运算 1 个、二元运算 2 个。
用统一的名字 `a`/`b`/`c` 可以让 `dump_ast` 之类的通用代码少写很多分支。
**命名可以有含义，但字段集合要统一**——这是一种常见的折中。

**为什么 `op` 的类型是 `int` 而不是 `TokType`？**
因为 `TokType` 是词法层的类型，`Node.op` 存的却是**语法层的运算符**。
直接复用 `TokType` 完全够用（它就是个 `int` 枚举），写成 `int` 只是少写几个字符。
如果你想更严谨，可以写 `TokType op;`——功能上没区别。

**为什么 `text` 是 4096 字节的固定数组，而 `params[]` 是 `char *`？**
因为 `text` 存的是**从 token 里拷过来的副本**（见 5.6），
而 `params[]` 存的将是**指向别处的名字**，生命周期不同。
本步 `params` 还没用上，所以两种写法并存不会引起混淆。

### 5.3 节点池 + `memset`：为什么必需

```c
/* 节点池：所有 AST 节点都从这里分配，用完为止 */
static Node nodes[MAX_NODES];
static int  nnodes;

/* 创建一个节点。memset 清零非常重要：否则 switch 里没覆盖到的字段
   会是随机值，后面调试会非常痛苦 */
static Node *node(NodeType ty, Token t) {
    if (nnodes >= MAX_NODES) fatal_at(t, "语法树节点超出上限");
    Node *n = &nodes[nnodes++];
    memset(n, 0, sizeof(*n));
    n->type = ty;
    n->tok  = t;
    return n;
}
```

**节点池（arena / pool allocator）是什么？**
预先开一个 `static Node nodes[MAX_NODES]` 大数组，
需要新节点时就把数组里下一个空位拿来用，`nnodes` 加一。
用完整个数组就报"语法树节点超出上限"。

**为什么不用 `malloc`？** 三个原因：

1. **不用管释放。** `malloc` 的每个节点都要在某个地方 `free`，
   而 AST 是"一次性建好、然后只读"的结构，释放它毫无意义。
   程序结束时整个进程地址空间被回收，节点池自然消失。
2. **不用检查 `malloc` 失败。** 静态数组在 BSS 段，永远分配成功。
3. **快。** 数组里顺序分配，缓存友好。

代价是内存**不能复用**——解析 1000 个表达式就是 1000 棵独立的树全占着。
教学项目里每个程序只解析一次，完全无所谓。

**`memset(n, 0, sizeof(*n))` 为什么必需？**

这是本步最容易被忽略、但一不注意就会让你 debug 到怀疑人生的一行。

`nodes` 是**静态数组**，进程刚启动时它的内容是**全 0**。
那为什么还要 `memset`？因为**第 2 次、第 3 次……复用同一个数组槽位时**，
那个位置还留着上一次解析的旧数据！

一个**实测过**的例子（见第 8 节"坑 2"）：

```c
/* 第一次解析：f(1, 2, 3) —— N_CALL 节点的 count 被设成 3 */
/* 第二次解析：g()        —— 复用同一个槽位 */

/* 没有 memset 时： */
n->count 仍然是 3                    /* ← 旧数据！ */
n->items[0..2] 仍然是上次的三个孩子   /* ← 旧数据！ */

/* dump_ast 按 count 打印，于是 g() 打印出了 3 个参数 */
```

`memset` 把整块结构体清零，保证：

> **每个字段在被显式赋值之前，它的值都是"确定的"（0），而不是"上一次残留的值"。**

C 语言里"未初始化的局部变量"和"静态变量的初始值"都是**不定值（indeterminate）**，
读它们是未定义行为。`memset` 就是把这个坑填平——它让
**"槽位被复用"和"槽位是全新的"这两种情况表现得完全一样**。

**顺便：`Node` 里有 `char text[MAX_TEXT]`（4096 字节），所以 `memset` 一次要清 4KB 多。**
对于要创建上万个节点的大程序，这会有性能开销。
更好的做法是**用联合体（union）让互斥字段共享内存**，或者干脆把 `text` 改成指针。
本步选择简单——**教学代码优先要"行为可预测"，不是"跑得最快"**。

### 5.4 递归下降：怎么用"一层一个函数"表达优先级

**递归下降（recursive descent）**是一种把语法规则直接翻译成函数调用的解析方法。
核心思想只有一句：

> **语法规则里每一层优先级，对应一个函数。**

MiniLang 的表达式优先级从低到高是这样排的：

```text
优先级低 ──────────────────────────────────────────► 优先级高

  expression()
      │
      ▼
  logical_or()      ||
      │
      ▼
  logical_and()     &&
      │
      ▼
  equal()           ==  !=
      │
      ▼
  compare()         <  <=  >  >=
      │
      ▼
  add()             +  -
      │
      ▼
  mul()             *  /  %
      │
      ▼
  unary()           -x  +x  !x
      │
      ▼
  primary()         数字  变量  函数调用  (表达式)
```

**为什么高优先级要放在低优先级的"里面"？**
因为**结合性（associativity）**决定了树的形状。
`1 - 2 - 3` 应该解析成 `(1-2)-3`（左结合），树是：

```text
二元 MINUS
├── 二元 MINUS
│   ├── 数字 1
│   └── 数字 2
└── 数字 3
```

而不是 `1-(2-3)`。**左结合 = 左子树更深 = 高优先级的那层在"更里面"被先绑定。**

所以在 `mul()` 里解析 `*` 的右操作数时，要调用**更内层**的 `unary()`；
在 `add()` 里解析 `+` 的右操作数时，要调用 `mul()`。
**越靠内的函数，优先级越高。**

一个具体的推导。假设我们要解析 `1 + 2 * 3`：

```text
expression() → logical_or() → ... → add()
  add() 调用 mul() 解析左操作数
    mul() 调用 unary() → primary() 得到 数字 1
    mul() 看到下一个是 '+'，不是 '*'，返回 数字 1
  add() 看到 '+'，吃掉它
    add() 调用 mul() 解析右操作数
      mul() 调用 unary() → primary() 得到 数字 2
      mul() 看到 '*'，吃掉它，继续
        mul() 调用 unary() → primary() 得到 数字 3
        mul() 看到 EOF，返回 二元 STAR(2,3)
    add() 拿到 二元 STAR(2,3)，返回
  add() 组装出 二元 PLUS(1, 二元 STAR(2,3))
```

结果就是：

```text
二元 PLUS
├── 数字 1
└── 二元 STAR
    ├── 数字 2
    └── 数字 3
```

`+` 在外层，`*` 在内层——**内层先结合**，所以 `*` 优先。
树形和优先级完全对应上了。

### 5.5 `bin_level`：一个循环搞定整张优先级表

如果每一层都手写成 `while (match(T_PLUS) || match(T_MINUS)) {...}`，
6 层优先级就要写 6 遍几乎一样的代码。本步用**一个通用函数**消除了重复：

```c
/* 一层二元运算：左结合的循环 */
static Node *bin_level(Node *(*next)(void), const TokType *ops, int nops) {
    Node *left = next();
    for (;;) {
        int found = 0;
        for (int i = 0; i < nops; i++) {
            if (cur()->type == ops[i]) {
                Token t = *cur();
                pos++;
                Node *n = node(N_BINARY, t);
                n->op = t.type;
                n->a  = left;
                n->b  = next();
                left  = n;
                found = 1;
                break;
            }
        }
        if (!found) return left;
    }
}
```

然后每一层只写三行：

```c
static Node *mul(void) {           /* * / % */
    const TokType o[] = {T_STAR, T_SLASH, T_PERCENT};
    return bin_level(unary, o, 3);
}
static Node *add(void) {           /* + - */
    const TokType o[] = {T_PLUS, T_MINUS};
    return bin_level(mul, o, 2);
}
static Node *compare(void) {       /* < <= > >= */
    const TokType o[] = {T_LT, T_LE, T_GT, T_GE};
    return bin_level(add, o, 4);
}
static Node *equal(void) {         /* == != */
    const TokType o[] = {T_EQ, T_NE};
    return bin_level(compare, o, 2);
}
static Node *logical_and(void) {   /* && */
    const TokType o[] = {T_AND};
    return bin_level(equal, o, 1);
}
static Node *logical_or(void) {    /* || */
    const TokType o[] = {T_OR};
    return bin_level(logical_and, o, 1);
}
```

**逐点拆解：**

1. **`Node *(*next)(void)` 是"函数指针"。**
   `next` 是一个指向"无参数、返回 `Node *` 的函数"的指针。
   `bin_level(unary, o, 3)` 就是在说："这一层的下一层（更高优先级）用 `unary` 来解析"。
   这是 C 里**没有闭包、没有一等函数**时实现"传函数"的经典手法。

2. **`left = next();` 先拿到左操作数。** 这一句对应"一元表达式或更高优先级的表达式"。

3. **内层 `for` 遍历这一层的运算符集合。** `ops[]` 是这一层允许的所有 `TokType`。
   `nops` 是它的长度。

4. **`left = n;` 是左结合的关键。**
   每解析出一个运算符，就把"已经攒起来的东西"更新成新的左操作数。
   下一个运算符的右操作数就接在它后面。
   这自然产生左结合的树形。

5. **`found` 标志 + `break`。** 内层循环找到运算符后要跳出，
   然后回到外层 `for(;;)` 继续。因为一个表达式里可以有很多运算符
   （`1 + 2 - 3 + 4`），所以外层必须是无限循环。

6. **`if (!found) return left;`** 当前的 token 不是这一层的任何运算符，
   说明这一层到此结束，把攒好的结果返回给上一层。

**为什么这个设计这么好？**
看 `bin_level` 的参数就知道它有多通用：

```text
bin_level(下一层的函数, 这一层的运算符表, 表长度)
```

加一层新优先级（比如第 4 步的位运算 `&` `|` `^`），
只要**加一个 5 行的函数 + 在上一层的 `ops[]` 里加一个元素**，
`bin_level` 一个字都不用改。

**但它也有代价**，你要清楚：

- **优先级顺序不再由代码结构表达，而是由"谁调用谁"表达。**
  你必须记住 `mul` 调 `unary`、`add` 调 `mul`……调反了就全错。
  传统写法（每层一个 `while`）虽然啰嗦，但优先级一眼可见。
- **`bin_level` 里的 `next` 是"下一层"，不是"上一层"。** 名字有点误导，
  叫 `higher` 或 `sub` 会更清楚。这是本步代码里一个可以改进的地方。
- **编译器无法帮你检查优先级顺序。** 手写 6 个 `while` 的时候，
  你至少能通过"代码从上到下的顺序"看出优先级；现在必须数函数调用链。

### 5.6 `match` vs `expect`：两种"取 token"的语义

```c
/* 当前 token 是不是 t？是就吃掉它，返回 1；不是就什么都不做，返回 0 */
static int match(TokType t) {
    if (cur()->type == t) { pos++; return 1; }
    return 0;
}

/* 当前 token 必须是 t，否则报错并退出 */
static Token expect(TokType t, const char *what) {
    if (cur()->type != t)
        fatal_at(*cur(), "语法错误：%s，当前是 '%s'", what, cur()->text);
    return toks[pos++];
}
```

| | `match(t)` | `expect(t, what)` |
|---|---|---|
| 匹配成功 | `pos++`，返回 1 | `pos++`，**返回那个 token** |
| 匹配失败 | **什么都不做**，返回 0 | **报错并 `exit(1)`** |
| 用在哪 | "这里**可能**有"某个 token | "这里**必须**有"某个 token |
| 例子 | `match(T_COMMA)` 检查有没有逗号 | `expect(T_RP, "函数调用需要 ')'")` |

**典型用法对比：**

```c
/* 有逗号就继续，没有就结束（参数列表可以为空） */
if (cur()->type != T_RP) {
    do {
        n->items[n->count++] = expression();
    } while (match(T_COMMA));
}
expect(T_RP, "函数调用需要 ')'");
```

这里 `match(T_COMMA)` 是"试探"的：第一个参数后面**可能**有逗号，
有就继续解析下一个参数，没有就结束循环。
而 `expect(T_RP, ...)` 是"必须"的：参数列表结束**一定**是 `)`。

**`expect` 为什么返回 `Token` 而不是 `void`？**
因为有些地方需要那个 token 的内容。本步的 `expect` 调用点其实都没用返回值
（`expect(T_RP, ...)` 的结果被丢弃），但返回 `Token` 更通用——
比如第 5 步解析 `let x = 1;` 时，`expect(T_ID, "变量名")` 返回的 token 里就有变量名。

**`expect` 的错误信息为什么这么写？**

```c
fatal_at(*cur(), "语法错误：%s，当前是 '%s'", what, cur()->text);
```

- `what` 是**人类可读的期望**，比如 `"函数调用需要 ')'"`。
- `cur()->text` 是**实际看到的**。对普通 token 是原文（`;`、`2`、`let`）；
  对 EOF token 是 `"<文件结束>"`（第 1 步 `add_token` 里塞的）。
  所以你会看到 `错误 [2:1]: 语法错误：表达式需要 ')'，当前是 '<文件结束>'` 这种消息。

### 5.7 函数调用为什么在 `primary` 里处理

```c
if (match(T_ID)) {
    /* 标识符后面紧跟 '('，说明是函数调用 */
    if (match(T_LP)) {
        Node *n = node(N_CALL, t);
        snprintf(n->text, sizeof(n->text), "%s", t.text);
        if (cur()->type != T_RP) {
            do {
                if (n->count >= 128) fatal_at(*cur(), "函数参数过多");
                n->items[n->count++] = expression();
            } while (match(T_COMMA));
        }
        expect(T_RP, "函数调用需要 ')'");
        return n;
    }
    Node *n = node(N_VAR, t);
    snprintf(n->text, sizeof(n->text), "%s", t.text);
    return n;
}
```

**为什么不在语法分析阶段就区分"变量"和"函数调用"？**
因为**词法上没法区分**：`f` 和 `x` 都是 `T_ID`。
唯一的区别是**后面跟不跟一个 `(`**——这是语法（syntactic）层面的信息，
只能在语法分析阶段判断。这叫 **"上下文无关"** 的经典例子：
同一个 token 序列前缀，配上不同的后继，语义就不同。

**为什么放在 `primary`（最底层）而不是单独一层？**
因为**函数调用的实参是一个完整的表达式**，它的优先级比所有运算符都高。
`f(1) + 2` 里的 `f(1)` 必须先绑定成一个整体。
所以函数调用属于"原子"（primary / atom）这一层，和数字、变量、括号同级。

**`if (cur()->type != T_RP)` 这个判断是干什么的？**
处理**零个参数**的调用 `f()`。如果直接写 `do { ... } while (match(T_COMMA));`，
那 `f()` 会在 `)` 上尝试解析一个表达式，然后报错。
所以要先看看下一个是不是 `)`，是的话说明参数列表为空。

**为什么 `n->count >= 128` 要检查？**
因为 `items` 是固定 128 个元素的数组。写多了就是**缓冲区溢出**。
这个检查和 `add_token` 里的 `ntok >= MAX_TOKENS` 是同一个思路。

### 5.8 拷贝名字：`memcpy` 还是 `snprintf`？

本步有三处需要把 token 里的名字拷进 `Node.text`：

```c
snprintf(n->text, sizeof(n->text), "%s", t.text);
```

而第 1 步的词法分析器用的是：

```c
size_t n = i - start;
if (n >= sizeof(b)) fatal_at(..., "标识符太长");
memcpy(b, src + start, n); b[n] = 0;
```

**为什么不一样？** 因为**风险不同**：

| | 词法分析器 | 语法分析器 |
|---|---|---|
| 数据来源 | 用户源码里的原始字节 | 已经是 `Token.text` 里的合法 C 字符串 |
| 长度 | **不可控**（用户可以写一个 10000 字符的标识符） | **必然 ≤ 4095**（`Token.text` 是 4096 字节数组，`add_token` 用 `snprintf` 拷进去的） |
| 需要检查长度吗 | **需要**，否则 `memcpy` 越界 | 不需要，源已经是安全的 |
| 用什么 | `memcpy` + 显式长度检查 | `snprintf`，简单安全 |

**`snprintf` 的三个好处：**

1. **自动截断**，永不越界——即使源比目标长。
2. **自动加 `'\0'`**。
3. **不依赖源字符串里没有嵌入的 `'\0'`**。
   实际上 `Token.text` 里不会有嵌入的 `'\0'`（`add_token` 用 `%s` 拷的），
   但 `snprintf` 不需要你操心这一点。

**为什么词法分析器不用 `snprintf` 而用 `memcpy`？**
因为 `snprintf` **静默截断**：一个 5000 字符的标识符会被悄悄砍成 4095 个字符，
没有任何提示。你会得到一个"名字莫名其妙变了"的程序。
`memcpy` + 长度检查则是**显式失败**：直接报"标识符太长"，告诉用户真相。

> 💡 这就是 `snprintf` 的一个常被忽略的坑：**它不报错，它只截断。**
> 本教程后面第 9 步做字符串模板时还会遇到这个问题的另一个版本。

**为什么 `Node.text` 要拷贝一份，而不是直接存 `Token.text` 的指针？**

因为 `Token.text` 是**数组**，不是指针。`Token` 里有 `char text[MAX_TEXT]`，
所以 `t.text` 是一个数组，**不能直接赋给 `char *`**。
如果 `Node` 想存一个 `const char *` 指向它，得写 `n->text = t.text;` ——这是**类型错误**，
`char[4096]` 不能隐式转成 `char *`（数组会退化成指针，但那是 `const char *`，
而 `n->text` 是可写数组，不是指针）。

要存指针就得把 `Node.text` 改成 `const char *name;`，然后
`n->name = t.text;`。这样**零拷贝**，但代价是：
`Node` 的生命周期必须依附于 `Token` 数组（而 `Token` 数组是静态的，所以其实也行）。

本步选择**拷贝**，理由：

- `Node.text` 同时要装三种东西（变量名 / 函数名 / 字符串内容），
  用一个定长数组最省事，不用搞三种生命周期。
- 拷贝一次的成本可以忽略（名字都很短）。
- 第 13 步模块化重构时，`Node` 会搬进自己的模块，
  拷贝让它不依赖 `Token` 的内部布局。

### 5.9 打印语法树：`dump_ast` 的连接线怎么画

```c
static void dump_ast(Node *n, const char *prefix) {
    if (!n) return;

    Node *kids[128];
    int   m = 0;

    switch (n->type) {
        case N_NUM: printf("数字 %g\n", n->num); return;
        case N_STR: printf("字符串 \"%s\"\n", n->text); return;
        case N_VAR: printf("变量 %s\n", n->text); return;
        case N_UNARY:
            printf("一元 %s\n", type_name((TokType)n->op));
            kids[m++] = n->a;
            break;
        case N_BINARY:
            printf("二元 %s\n", type_name((TokType)n->op));
            kids[m++] = n->a;
            kids[m++] = n->b;
            break;
        case N_CALL:
            printf("调用 %s\n", n->text);
            for (int i = 0; i < n->count; i++) kids[m++] = n->items[i];
            break;
    }

    for (int i = 0; i < m; i++) {
        int last = (i == m - 1);
        char sub[256];
        snprintf(sub, sizeof(sub), "%s%s", prefix, last ? "    " : "│   ");
        printf("%s%s", prefix, last ? "└── " : "├── ");
        dump_ast(kids[i], sub);
    }
}
```

**核心技巧是 `prefix`（前缀）逐层往下传。**
每个递归调用带着"我这一层的缩进前缀"，打印孩子时：
- 孩子**不是**最后一个 → 用 `├── `，并且下一层的前缀要加上 `│   `（竖线 + 3 空格）
- 孩子**是**最后一个 → 用 `└── `，下一层的前缀要加上 4 个空格（**没有竖线**）

这就是 ASCII 树形图的画法。用一个例子走一遍：

```text
二元 PLUS          prefix = ""
├── 数字 1          prefix = "│   "  (不是最后一个)
└── 二元 STAR      prefix = "    "  (是最后一个)
    ├── 数字 2      prefix = "    │   "
    └── 数字 3      prefix = "        "
```

`sub` 缓冲区 256 字节，足够放十几层缩进。层数再深就该用堆分配了，
但本步的表达式嵌套深度也就几十层。

**注意 `type_name((TokType)n->op)` 这个强制转换。**
`n->op` 的类型是 `int`，`type_name` 要的是 `TokType`（枚举）。
C 里枚举和整数可以隐式互转，但**显式转换更清晰**，而且
`gcc -Wall -Wextra` 对某些隐式转换会报警。

### 5.10 完整的运算符优先级表

这是本步最重要的产出。从**低**到**高**：

| 优先级 | 层 | 运算符 | 结合性 | 解析函数 | 树形 |
|---|---|---|---|---|---|
| 1（最低） | 逻辑或 | `\|\|` | 左 | `logical_or()` | 最外层 |
| 2 | 逻辑与 | `&&` | 左 | `logical_and()` | |
| 3 | 相等 | `==` `!=` | 左 | `equal()` | |
| 4 | 比较 | `<` `<=` `>` `>=` | 左 | `compare()` | |
| 5 | 加减 | `+` `-` | 左 | `add()` | |
| 6 | 乘除模 | `*` `/` `%` | 左 | `mul()` | |
| 7 | 一元 | `-x` `+x` `!x` | 右 | `unary()` | |
| 8（最高） | 原子 | `123` `x` `f(...)` `(expr)` | — | `primary()` | 最内层 |

**从高到低的记忆口诀**：`括号 > 一元 > 乘除 > 加减 > 比较 > 相等 > 与 > 或`。

**对应到代码**（"下一层"就是优先级更高的一层）：

```text
logical_or()   → bin_level(logical_and, {T_OR},          1)
logical_and()  → bin_level(equal,       {T_AND},         1)
equal()        → bin_level(compare,     {T_EQ,T_NE},     2)
compare()      → bin_level(add,         {T_LT,T_LE,T_GT,T_GE}, 4)
add()          → bin_level(mul,         {T_PLUS,T_MINUS},2)
mul()          → bin_level(unary,       {T_STAR,T_SLASH,T_PERCENT}, 3)
unary()        → primary()
```

**第 4 步会激活比较和逻辑运算。** 那时候你会发现：
`a < b == c < d` 会被解析成 `(a<b) == (c<d)` 而不是 `a < (b == (c<d))`——
因为优先级表已经定死了。

**二元运算符全部左结合**，一元运算符右结合（`- -x` 是 `-(-x)`）。
`unary()` 里递归调用自己就实现了右结合：

```c
n->a = unary();   /* 不是 primary()，所以 - - x 能正确嵌套 */
```

---

## 6. 动手：把代码加进来

### 6.1 改动清单

| 位置锚点 | 改了什么 | 大概多少行 | 为什么 |
|---|---|---|---|
| `#define` 区 | 加 `MAX_NODES 20000`、`MAX_PARAMS 32` | 2 | 节点池容量、函数参数上限 |
| 全局变量区 | 加 `static int pos;` | 1 | 语法分析器的游标（第 1 步的 `ntok` 是"总数"，方向相反） |
| `type_name` 注释 | 改一个字 | 1 | 注释里"中文/英文"的说法不准确了，实际返回的是英文名 |
| AST 区块（插在 `type_name` 之后） | `NodeType` 枚举 + `Node` 结构体 + 节点池 + `node()` | 约 50 | 见 5.2 / 5.3 |
| 语法分析区块 | `expression` 前向声明、`cur()`、`match()`、`expect()`、`primary()`、`unary()`、`bin_level()`、6 个优先级层函数 | 约 120 | 见 5.4–5.7 |
| 打印区块（插在 `read_file` 之前） | `dump_ast()` | 约 40 | 见 5.9 |
| `mode_ast()` | 填上真正的实现 | 8 | 见下 |
| `mode_run()` | 改一句提示文字 | 1 | 提一下现在也能用 `ast` 模式了 |

### 6.2 分段讲解

#### 第 1 段：全局游标 `pos`

```c
static Token toks[MAX_TOKENS];   /* token 数组 */
static int   ntok;               /* 已经识别了多少个 token */
static int   pos;                /* 语法分析器当前读到第几个 token */
```

`ntok` 和 `pos` 的区别：

| | 含义 | 谁改它 |
|---|---|---|
| `ntok` | 一共识别了多少个 token | `add_token`（词法分析阶段） |
| `pos` | 语法分析器当前**读到**第几个 | `match` / `expect` / `primary`（语法分析阶段） |

两个阶段是**先后**的：先 `lex()` 把 `ntok` 填满，再开始用 `pos` 走。
第 1 步只有 `ntok`，因为还没有"读"这个动作。

#### 第 2 段：`cur()` / `match()` / `expect()`

```c
static Token *cur(void) { return &toks[pos]; }
```

`cur()` 返回**当前 token 的指针**。注意它返回的是 `&toks[pos]`——
**指向数组内部的指针**，不是拷贝。

那为什么 `expect()` 里写 `*cur()`（解引用拷贝）？
因为 `fatal_at` 要一个 `Token` **值**。`*cur()` 拷贝一份 4096 多字节的结构体
（因为 `Token.text` 是 4096 字节数组），有点浪费，但简单且安全。
如果你想避免拷贝，可以把 `fatal_at` 改成收 `const Token *`——但那样
INV-7 的签名就变了，所以本步**保持原样**。

**`match()` 和 `expect()` 的完整实现见 5.6。**

一个容易忽略的细节：`match()` 里 `pos++` 之后**立刻 return**，
所以下面 `return 0;` 那句只在**没匹配**时执行。
如果写成：

```c
static int match(TokType t) {
    if (cur()->type == t) { pos++; return 1; }
    return 0;
}
```

这是对的。但如果你写成 `if (...) pos++; return cur()->type == t;` 就**错了**——
因为 `pos++` 之后 `cur()` 已经指向下一个 token 了。

#### 第 3 段：`primary()`——原子表达式

```c
static Node *primary(void) {
    Token t = *cur();

    if (match(T_NUM)) {
        Node *n = node(N_NUM, t);
        n->num = t.num;
        return n;
    }
    ...
    fatal_at(t, "语法错误：此处需要数字、变量、函数调用或括号表达式");
    return NULL;
}
```

**为什么开头要 `Token t = *cur();`？**
因为 `match(T_NUM)` 会 `pos++`，之后 `cur()` 就指向下一个 token 了。
必须**在 match 之前**把当前 token 拍个快照（和第 1 步 `int line = src_line, col = src_col;` 一模一样的思路）。

**`Token t = *cur();` 拷贝了 4096 多字节，会不会太慢？**
在这个规模下无所谓。但更讲究的写法是只拷需要的字段：

```c
Token t = *cur();   /* 简单版 */
```

或者说，如果 `Node.tok` 不需要完整的 `text`，可以只存 `line`/`col`。
本步选择存完整的 `Token`——因为第 12 步做错误定位时，
可能要显示 token 的原文来高亮。

**`fatal_at` 之后的 `return NULL;` 是给编译器看的。**
`fatal_at` 标了 `_Noreturn`，编译器**本来就知道**它不会返回，
所以那句 `return NULL;` 是多余的。但加上它能让**旧版编译器**（或者
关闭了 `_Noreturn` 分析的情况）不报 "control reaches end of non-void function"。
这是防御性的好习惯。

**括号表达式的处理：**

```c
if (match(T_LP)) {
    Node *n = expression();
    expect(T_RP, "表达式需要 ')'");
    return n;
}
```

括号里的内容直接**递归调用 `expression()`**（最外层入口），所以
`(1 + 2) * 3` 里的括号能装下任意复杂的表达式。
返回的是**括号里面那棵子树本身**——括号在 AST 里**不留下痕迹**，
它只影响树的形状。这是"抽象"语法树（Abstract Syntax Tree）里"抽象"二字的含义：
括号、分号这些纯语法的东西被丢掉了，只保留语义结构。

#### 第 4 段：`unary()` 和 `bin_level()`

```c
/* 一元运算：-x +x !x */
static Node *unary(void) {
    if (cur()->type == T_MINUS || cur()->type == T_PLUS || cur()->type == T_NOT) {
        Token t = *cur();
        pos++;
        Node *n = node(N_UNARY, t);
        n->op = t.type;
        n->a  = unary();     /* 递归调用自己 → 右结合 */
        return n;
    }
    return primary();
}
```

**为什么一元用 `recursion` 而不是 `bin_level`？**
因为一元运算符只有**一个**操作数，而且要**右结合**：
`- - x` = `-(-x)`。如果写成循环 `while (match(T_MINUS)) { ... }`，
操作数会在循环结束后才解析，形状就变成 `(-(-x))`……

等等，仔细看：这里 `n->a = unary()` 是在**吃掉运算符之后立刻递归**，
所以 `- - x` 解析过程是：
- 吃掉第一个 `-`，创建节点 n1，`n1->a = unary()`
- `unary()` 看到第二个 `-`，吃掉，创建节点 n2，`n2->a = unary()`
- `unary()` 看到 `x`，返回 `变量 x`
- n2 = 一元 MINUS(x)，返回
- n1 = 一元 MINUS(n2) = 一元 MINUS(一元 MINUS(x))

结果：`--x` 的树是 `一元 MINUS → 一元 MINUS → 变量 x`，**右结合，正确**。

> 💡 注意 `unary()` 里没有用 `match()`，而是手写了
> `if (cur()->type == T_MINUS || ...) { Token t = *cur(); pos++; ... }`。
> 因为这里需要**先保存 token 再前进**，而 `match()` 只返回 0/1 不返回 token。
> 这就是为什么 `expect()` 要返回 `Token`——它是"必须匹配 + 给我 token"的版本。

**`bin_level()` 见 5.5。**

#### 第 5 段：`mode_ast()` 填上真正的实现

```c
static int mode_ast(const char *path) {
    src = read_file(path);
    lex();
    Node *e = expression();
    if (cur()->type != T_EOF)
        fatal_at(*cur(), "语法错误：表达式结束后还有多余的 '%s'", cur()->text);
    printf("语法树：\n");
    dump_ast(e, "");
    free((void *)src);
    return had_error ? 1 : 0;
}
```

四个要点：

1. **`lex()` 一定要先调用。** 语法分析器读的是 `toks[]`，
   词法分析器负责把它填满。
2. **`expression()` 只解析一个表达式。** 本步的 `ast` 模式不支持语句。
3. **检查"多余的 token"。** 解析完一个表达式后，如果当前 token 不是 `T_EOF`，
   说明源码里还有别的东西。**这个检查不能省**——
   否则 `1 2` 会被当成合法的 `1`，`2` 被无声地丢掉。
4. **`free((void *)src);`** `src` 是 `const char *`，`free` 要 `void *`，
   所以要显式转换。（C 里 `free` 的参数类型是 `void *`，
   把 `const char *` 直接传过去会报 "discards 'const' qualifier" 警告。）

**`mode_run` 的提示文字也要改：**

```diff
-    fprintf(stderr, "解释执行尚未实现，请先用 tokens 模式查看词法结果\n");
+    fprintf(stderr, "解释执行尚未实现，请先用 tokens 或 ast 模式查看结果\n");
```

> ⚠️ **这是本步唯一一处"修改既有行"**（除了那句注释）。
> 按本教程的 INV-8，新代码原则上只插入不修改。
> 这里改的是一句**面向用户的提示文字**（不是逻辑），
> 而且改的理由很实际：现在 `ast` 模式真的能用了，
> 提示不跟着改就会误导读者。这是白名单式的例外，说明一下即可。

### 6.3 完整 diff

> **怎么用这段 diff**：推荐**照着敲**——每改一小段就编译一次，比一次性粘贴更容易记住。
> 如果想直接应用，把下面的内容存成 `step.patch`，然后在**上一步的目录**里执行：
>
> ```sh
> patch -p1 < ../002/step.patch
> ```
>
> 应用完和本目录的 `minilang.c` **逐字节一致**（可以 `diff` 验证）。

<details>
<summary>点开看完整 diff（真实 diff -u 输出）</summary>

```diff
--- a/minilang.c
+++ b/minilang.c
@@ -27,7 +27,9 @@
 #include <stdarg.h>
 
 #define MAX_TOKENS 20000
+#define MAX_NODES  20000
 #define MAX_TEXT   4096
+#define MAX_PARAMS 32
 
 /* ------------------------------------------------------------------ */
 /* Token：词法分析器产出的最小单位                                     */
@@ -50,6 +52,7 @@
 
 static Token toks[MAX_TOKENS];   /* token 数组 */
 static int   ntok;               /* 已经识别了多少个 token */
+static int   pos;                /* 语法分析器当前读到第几个 token */
 static const char *src;          /* 整份源代码 */
 static int   src_line, src_col;  /* 扫描器当前所在的位置 */
 static int   had_error;          /* 出过错没有 */
@@ -214,7 +217,7 @@
     add_token(T_EOF, "<文件结束>", 0, src_line, src_col);
 }
 
-/* token 类型的中文/英文名字，打印时用 */
+/* token 类型的名字，打印时用 */
 static const char *type_name(TokType t) {
     switch (t) {
         case T_EOF:     return "EOF";
@@ -250,9 +253,220 @@
         case T_NOT:     return "NOT";
     }
     return "?";
+}
+
+/* ------------------------------------------------------------------ */
+/* AST：抽象语法树                                                     */
+/* ------------------------------------------------------------------ */
+typedef enum {
+    N_NUM,      /* 数字字面量        */
+    N_STR,      /* 字符串字面量      */
+    N_VAR,      /* 变量引用          */
+    N_UNARY,    /* 一元运算：-x !x   */
+    N_BINARY,   /* 二元运算：a + b   */
+    N_CALL      /* 函数调用：f(a,b)  */
+} NodeType;
+
+typedef struct Node Node;
+struct Node {
+    NodeType type;
+    Token   tok;              /* 出错定位用：行列号 */
+    double  num;              /* N_NUM 的数值      */
+    char    text[MAX_TEXT];   /* N_VAR/N_CALL 的名字，N_STR 的内容 */
+    int     op;               /* 运算符的 TokType  */
+    Node   *a, *b, *c;        /* 子节点            */
+    Node   *items[128];       /* N_CALL 的实参列表 */
+    int     count;
+    char   *params[MAX_PARAMS]; /* 函数参数名（后面几步才用） */
+    int     nparams;
+};
+
+/* 节点池：所有 AST 节点都从这里分配，用完为止 */
+static Node nodes[MAX_NODES];
+static int  nnodes;
+
+/* 创建一个节点。memset 清零非常重要：否则 switch 里没覆盖到的字段
+   会是随机值，后面调试会非常痛苦 */
+static Node *node(NodeType ty, Token t) {
+    if (nnodes >= MAX_NODES) fatal_at(t, "语法树节点超出上限");
+    Node *n = &nodes[nnodes++];
+    memset(n, 0, sizeof(*n));
+    n->type = ty;
+    n->tok  = t;
+    return n;
+}
+
+/* ------------------------------------------------------------------ */
+/* 语法分析器：递归下降                                                 */
+/* ------------------------------------------------------------------ */
+static Node *expression(void);
+
+static Token *cur(void) { return &toks[pos]; }
+
+/* 当前 token 是不是 t？是就吃掉它，返回 1；不是就什么都不做，返回 0 */
+static int match(TokType t) {
+    if (cur()->type == t) { pos++; return 1; }
+    return 0;
+}
+
+/* 当前 token 必须是 t，否则报错并退出 */
+static Token expect(TokType t, const char *what) {
+    if (cur()->type != t)
+        fatal_at(*cur(), "语法错误：%s，当前是 '%s'", what, cur()->text);
+    return toks[pos++];
+}
+
+/* 最基础的东西：数字、字符串、变量、函数调用、括号表达式 */
+static Node *primary(void) {
+    Token t = *cur();
+
+    if (match(T_NUM)) {
+        Node *n = node(N_NUM, t);
+        n->num = t.num;
+        return n;
+    }
+    if (match(T_STR)) {
+        Node *n = node(N_STR, t);
+        snprintf(n->text, sizeof(n->text), "%s", t.text);
+        return n;
+    }
+    if (match(T_ID)) {
+        /* 标识符后面紧跟 '('，说明是函数调用 */
+        if (match(T_LP)) {
+            Node *n = node(N_CALL, t);
+            snprintf(n->text, sizeof(n->text), "%s", t.text);
+            if (cur()->type != T_RP) {
+                do {
+                    if (n->count >= 128) fatal_at(*cur(), "函数参数过多");
+                    n->items[n->count++] = expression();
+                } while (match(T_COMMA));
+            }
+            expect(T_RP, "函数调用需要 ')'");
+            return n;
+        }
+        Node *n = node(N_VAR, t);
+        snprintf(n->text, sizeof(n->text), "%s", t.text);
+        return n;
+    }
+    if (match(T_LP)) {
+        Node *n = expression();
+        expect(T_RP, "表达式需要 ')'");
+        return n;
+    }
+
+    fatal_at(t, "语法错误：此处需要数字、变量、函数调用或括号表达式");
+    return NULL;
+}
+
+/* 一元运算：-x +x !x */
+static Node *unary(void) {
+    if (cur()->type == T_MINUS || cur()->type == T_PLUS || cur()->type == T_NOT) {
+        Token t = *cur();
+        pos++;
+        Node *n = node(N_UNARY, t);
+        n->op = t.type;
+        n->a  = unary();
+        return n;
+    }
+    return primary();
 }
 
+/* 一层二元运算：左结合的循环 */
+static Node *bin_level(Node *(*next)(void), const TokType *ops, int nops) {
+    Node *left = next();
+    for (;;) {
+        int found = 0;
+        for (int i = 0; i < nops; i++) {
+            if (cur()->type == ops[i]) {
+                Token t = *cur();
+                pos++;
+                Node *n = node(N_BINARY, t);
+                n->op = t.type;
+                n->a  = left;
+                n->b  = next();
+                left  = n;
+                found = 1;
+                break;
+            }
+        }
+        if (!found) return left;
+    }
+}
+
+static Node *mul(void) {           /* * / % */
+    const TokType o[] = {T_STAR, T_SLASH, T_PERCENT};
+    return bin_level(unary, o, 3);
+}
+static Node *add(void) {           /* + - */
+    const TokType o[] = {T_PLUS, T_MINUS};
+    return bin_level(mul, o, 2);
+}
+static Node *compare(void) {       /* < <= > >= */
+    const TokType o[] = {T_LT, T_LE, T_GT, T_GE};
+    return bin_level(add, o, 4);
+}
+static Node *equal(void) {         /* == != */
+    const TokType o[] = {T_EQ, T_NE};
+    return bin_level(compare, o, 2);
+}
+static Node *logical_and(void) {   /* && */
+    const TokType o[] = {T_AND};
+    return bin_level(equal, o, 1);
+}
+static Node *logical_or(void) {    /* || */
+    const TokType o[] = {T_OR};
+    return bin_level(logical_and, o, 1);
+}
+
+/* 表达式的最外层入口 */
+static Node *expression(void) { return logical_or(); }
+
 /* ------------------------------------------------------------------ */
+/* 打印语法树                                                          */
+/* ------------------------------------------------------------------ */
+/* 把一棵子树打印成带连接线的树。
+   prefix 是"给子节点用的缩进前缀"，逐层往下加 "│   " 或 "    " */
+static void dump_ast(Node *n, const char *prefix) {
+    if (!n) return;
+
+    Node *kids[128];
+    int   m = 0;
+
+    switch (n->type) {
+        case N_NUM:
+            printf("数字 %g\n", n->num);
+            return;
+        case N_STR:
+            printf("字符串 \"%s\"\n", n->text);
+            return;
+        case N_VAR:
+            printf("变量 %s\n", n->text);
+            return;
+        case N_UNARY:
+            printf("一元 %s\n", type_name((TokType)n->op));
+            kids[m++] = n->a;
+            break;
+        case N_BINARY:
+            printf("二元 %s\n", type_name((TokType)n->op));
+            kids[m++] = n->a;
+            kids[m++] = n->b;
+            break;
+        case N_CALL:
+            printf("调用 %s\n", n->text);
+            for (int i = 0; i < n->count; i++) kids[m++] = n->items[i];
+            break;
+    }
+
+    for (int i = 0; i < m; i++) {
+        int last = (i == m - 1);
+        char sub[256];
+        snprintf(sub, sizeof(sub), "%s%s", prefix, last ? "    " : "│   ");
+        printf("%s%s", prefix, last ? "└── " : "├── ");
+        dump_ast(kids[i], sub);
+    }
+}
+
+/* ------------------------------------------------------------------ */
 /* 读文件                                                              */
 /* ------------------------------------------------------------------ */
 static char *read_file(const char *path) {
@@ -302,14 +516,20 @@
 }
 
 static int mode_ast(const char *path) {
-    (void)path;
-    fprintf(stderr, "语法分析尚未实现，请先用 tokens 模式查看词法结果\n");
-    return 1;
+    src = read_file(path);
+    lex();
+    Node *e = expression();
+    if (cur()->type != T_EOF)
+        fatal_at(*cur(), "语法错误：表达式结束后还有多余的 '%s'", cur()->text);
+    printf("语法树：\n");
+    dump_ast(e, "");
+    free((void *)src);
+    return had_error ? 1 : 0;
 }
 
 static int mode_run(const char *path) {
     (void)path;
-    fprintf(stderr, "解释执行尚未实现，请先用 tokens 模式查看词法结果\n");
+    fprintf(stderr, "解释执行尚未实现，请先用 tokens 或 ast 模式查看结果\n");
     return 1;
 }
 
```

</details>

### 6.4 插入位置说明

本步有 **5 处插入 + 2 处修改**，位置如下：

```text
① #define 区
   MAX_TOKENS 20000
   MAX_NODES  20000        ← 插入
   MAX_TEXT   4096
   MAX_PARAMS 32           ← 插入

② 全局变量区
   static int   ntok;
   static int   pos;       ← 插入
   static const char *src;

③ type_name() 上方那行注释
   ← 改一个字（"中文/英文名字" → "名字"）

④ type_name() 函数结束后、"读文件"注释块之前
   ← 插入两大块：
      /* AST：抽象语法树 */  (NodeType, Node, nodes, nnodes, node())
      /* 语法分析器：递归下降 */  (expression 声明, cur, match, expect,
                                   primary, unary, bin_level, mul/add/compare/
                                   equal/logical_and/logical_or, expression)
      /* 打印语法树 */  (dump_ast)

⑤ read_file() 之后、print_escaped() 之前 —— 无需插入（dump_ast 已在上一步的位置）

⑥ mode_ast() 函数体
   ← 整段替换（从 `(void)path;` 那三行改成真正的实现）

⑦ mode_run() 里那行 fprintf
   ← 改一句话
```

**为什么 AST 区块放在 `type_name()` 之后？**
因为 `dump_ast()` 要调用 `type_name()`。C 要求先声明后使用，
所以 `type_name()` 必须在 `dump_ast()` 前面。

**为什么语法分析器放在 `dump_ast()` 前面？**
因为 `dump_ast()` 的参数是 `Node *`，要 `Node` 类型先定义。
而 `Node` 类型又在 AST 区块里。所以顺序是：
**AST 定义 → 语法分析器 → dump_ast**，一条依赖链。

**为什么 `mode_ast()` 里 `lex()` 在 `expression()` 之前？**
运行时顺序，不是代码位置顺序。但**代码位置**上 `mode_ast` 在文件很后面，
所以 `lex()`（在文件前面）当然已经定义好了。

## 7. 测试

`./run.sh` 的真实输出：

```text
▶ 用例（模式：ast）
  ✅ case_01_precedence.ml        [ast]
  ✅ case_02_parentheses.ml       [ast]
  ✅ case_03_call.ml              [ast]
  ✅ case_04_unary.ml             [ast]
  ✅ case_05_missing_operand.ml   [ast]
▶ demo.ml
  | 语法树：
  | 二元 PLUS
  | ├── 数字 1
  | └── 二元 STAR
  |     ├── 数字 2
  |     └── 数字 3

✅ 全部通过（6 项）
```

### 逐个用例说明

**case_01_precedence** —— 输入 `1 + 2 * 3`，期望 stdout：

```text
语法树：
二元 PLUS
├── 数字 1
└── 二元 STAR
    ├── 数字 2
    └── 数字 3
```

**为什么是这个形状？** `*` 的优先级比 `+` 高，所以 `2 * 3` 先结合，
变成 `+` 的右子树。回到 5.4 的推导：`add()` 调用 `mul()` 解析右操作数，
`mul()` 把 `2 * 3` 组装成一个 `N_BINARY` 节点返回给 `add()`。

**case_02_parentheses** —— 输入 `(1 + 2) * 3`，期望 stdout：

```text
语法树：
二元 STAR
├── 二元 PLUS
│   ├── 数字 1
│   └── 数字 2
└── 数字 3
```

**括号改变了树形。** 对比 case_01：同样两个运算符，位置完全颠倒了。
这正是"括号在 AST 里不留痕迹，但改变树形"的含义——
括号本身没有对应的节点类型，但它的**作用**（改变结合顺序）被保留在了树里。

**case_03_call** —— 输入 `f(1, 2 + 3)`，期望 stdout：

```text
语法树：
调用 f
├── 数字 1
└── 二元 PLUS
    ├── 数字 2
    └── 数字 3
```

**函数调用的实参是完整表达式**，所以 `2 + 3` 整棵树作为一个参数。
实参按顺序放在 `items[]` 里，`dump_ast` 按顺序打印成孩子。
注意 `调用 f` 下面**直接**是参数，没有额外的"参数列表"中间层——
`N_CALL` 的孩子就是参数本身。

**case_04_unary** —— 输入 `-x * !y`，期望 stdout：

```text
语法树：
二元 STAR
├── 一元 MINUS
│   └── 变量 x
└── 一元 NOT
    └── 变量 y
```

**一元运算符的优先级高于 `*`。** 所以 `-x` 先结合成一个整体，
再和 `!y` 一起作为 `*` 的两个操作数。
注意 `一元 MINUS` 只有一个孩子（`变量 x`），
`一元 NOT` 也只有一个孩子（`变量 y`）——和 `二元` 节点有两个孩子形成对比。

**case_05_missing_operand** —— 输入 `1 +`（文件末尾有换行），期望 stderr：

```text
错误 [2:1]: 语法错误：此处需要数字、变量、函数调用或括号表达式
```

期望退出码 `1`。

**为什么报在 `[2:1]` 而不是 `[1:4]`？**
因为 `1 +` 后面 `primary()` 被调用，此时 `cur()` 指向 **EOF token**。
EOF token 的行列号是"源码结束处"——文件 `1 +\n` 的最后一行是第 2 行（空行），
所以 EOF 在 `[2:1]`。

**这正是第 1 步给 `Token` 存 `line`/`col` 的价值**：连"源码结束了"这种错误
都能报出准确位置。

### 其他真实运行

**多余的 token**（`1 2`）：

```console
$ printf '1 2\n' > /tmp/x.ml
$ ./minilang /tmp/x.ml ast
错误 [1:3]: 语法错误：表达式结束后还有多余的 '2'
```

**缺右括号**（`(1 + 2`）：

```console
$ printf '(1 + 2\n' > /tmp/y.ml
$ ./minilang /tmp/y.ml ast
错误 [2:1]: 语法错误：表达式需要 ')'，当前是 '<文件结束>'
```

**零参数调用**（`f()`）：

```console
$ printf 'f()\n' > /tmp/z.ml
$ ./minilang /tmp/z.ml ast
语法树：
调用 f
```

（`调用 f` 下面没有孩子，因为实参个数是 0。）

**`run` 模式还没实现**：

```console
$ ./minilang demo.ml
解释执行尚未实现，请先用 tokens 或 ast 模式查看结果
```

**`tokens` 模式完全没变**（第 1 步的词法器一行没动，INV-1 成立）：

```console
$ printf '1 + 2 * 3\n' > /tmp/e.ml
$ ./minilang /tmp/e.ml tokens
[1:1] NUM      1  (1)
[1:3] PLUS     + 
[1:5] NUM      2  (2)
[1:7] STAR     * 
[1:9] NUM      3  (3)
[2:1] EOF      <文件结束>
```

## 8. 易错点与坑

### 坑 1：`primary()` 里 `Token t = *cur()` 写晚了

**现象**：所有错误信息的行列号都指向"下一个" token，差一位。

**原因**：`match(T_NUM)` 会 `pos++`，之后 `cur()` 指向的是**下一个** token。

**正确写法**：

```c
static Node *primary(void) {
    Token t = *cur();      /* ← 必须在任何 match 之前 */
    if (match(T_NUM)) { ... }
```

**自检方法**：故意写一个语法错误，看报出来的行列号是不是那个真正出错的位置。
本步 `case_05_missing_operand` 就是这么定位的。

### 坑 2：`node()` 里忘了 `memset`

**现象**：单个表达式完全正常，但一旦**节点池的某个槽位被复用**，
那个节点就会带上上一次的数据——表现出来的可能是"参数个数不对"、
"字符串内容莫名其妙多了几个字符"、"多出几个孩子"。

**原因**：`nodes` 是静态数组，**进程刚启动时**全是 0，
所以**第一次**用某个槽位看起来完全正常。
但只要**同一个槽位被第二次使用**（比如一个程序里解析多个表达式、
或者节点池做了复用），那个位置就还留着上一次解析的旧数据。

这个 bug 极其阴险：**单表达式测试全过**，出问题的地方离现场很远。

**正确写法**：`memset(n, 0, sizeof(*n));` —— 无条件执行，不要"优化"掉。

**怎么主动验证**（下面这个例子是**实测**过的，不是想象的）：

在 `node()` 里注释掉 `memset`，然后让 `mode_ast` 解析**两个**表达式，
中间把 `nnodes` 归零（模拟节点池槽位被复用）：

```c
Node *e = expression();
nnodes = 0;              /* 模拟槽位复用 */
e = expression();
```

输入 `f(1, 2, 3)` 和 `g()`：

```text
去掉 memset（错误）：        有 memset（正确）：
语法树：                    语法树：
调用 g                      调用 g
├── 数字 1                  （没有孩子）
├── 数字 2
└── 数字 3
```

**`g()` 本来没有参数，却打印出了 3 个参数！**
原因是 `N_CALL` 节点的 `count` 字段没有被清零，
残留了上一个 `f(1,2,3)` 的 `count = 3`，
于是 `dump_ast` 按 `count` 打印了 3 个**上一次解析留下的**孩子。

**这就是 `memset` 的价值**：它让"槽位被复用"和"槽位是全新的"
这两种情况表现得**完全一样**。

### 坑 3：优先级层级调反了

**现象**：`1 + 2 * 3` 算出来是 9 而不是 7（或者树形是 `PLUS(1, ...)` 变成 `STAR(PLUS(1,2), 3)`）。

**原因**：把 `mul()` 写成了调用 `add()`，或者把 `add()` 写成了调用 `unary()`。

**正确写法**：**越靠内优先级越高**：

```text
logical_or → logical_and → equal → compare → add → mul → unary → primary
```

也就是 `bin_level(X, ops, n)` 里的 `X` 必须是**下一层（更高优先级）**的函数。

**怎么验证**：跑 `tests/case_01_precedence.ml`，树的根必须是 `二元 PLUS`，
右孩子必须是 `二元 STAR`。如果根是 `二元 STAR`，说明层级反了。

### 坑 4：`bin_level` 里 `left = n` 写成了 `left = n->b`

**现象**：多个同类运算符时丢操作数。比如 `1 + 2 + 3` 只算得出 `2 + 3`。

**原因**：左结合的关键就是每轮都把**整个已完成的表达式**当成新的左操作数。

**正确写法**：

```c
n->a  = left;    /* 左操作数 = 之前攒的 */
n->b  = next();  /* 右操作数 = 更高优先级的那一层 */
left  = n;       /* ← 然后更新成新节点 */
```

顺序也很重要：必须**先** `n->a = left`，**再** `left = n`，
否则 `n->a` 会指向自己（形成环，求值时无限递归）。

### 坑 5：函数指针类型写错

**现象**：`warning: incompatible pointer types passing 'Node *(*)(void)' to parameter of type 'Node *(*)(void)'`（其实长得一样但报错）。

**原因**：如果 `unary` 的声明是 `static Node *unary(void)`，
而 `bin_level` 的参数写成 `Node *(*next)(void)`，这是对的。
但如果你写成 `Node *(*next)()`（省略参数列表），C 会认为它接受**任意参数**，
类型不兼容，编译报错或警告。

**正确写法**：参数列表要**完整**写出来：`Node *(*next)(void)`。

### 坑 6：`n->items[n->count++]` 越界

**现象**：参数超过 128 个时堆/栈溢出，可能直接崩溃。

**原因**：`items` 是固定 128 元素的数组。

**正确写法**：先检查再写：

```c
if (n->count >= 128) fatal_at(*cur(), "函数参数过多");
n->items[n->count++] = expression();
```

### 坑 7：`dump_ast` 里 `kids` 数组不够大

**现象**：参数超过 128 个时 `kids[m++]` 越界。

**原因**：`kids[128]` 和 `n->items[128]` 大小相同，
但 `kids` 的下标是 `m`，从 0 开始——**正好**能放下 128 个。
一旦 `n->count` 能超过 128，两者就都溢出了。

**正确写法**：用 `MAX_PARAMS` 之类的常量统一管理大小，
或者把 `kids` 改成动态分配。本步因为 `count` 已经有上限检查，所以安全。

### 坑 8：忘了检查"多余的 token"

**现象**：`1 2` 被当成合法的表达式 `1`，`2` 被无声地丢弃。

**原因**：`expression()` 返回后没有确认 token 流已经到末尾。

**正确写法**：

```c
if (cur()->type != T_EOF)
    fatal_at(*cur(), "语法错误：表达式结束后还有多余的 '%s'", cur()->text);
```

### 坑 9：`free((void *)src)` 的 const 转换

**现象**：编译警告 `passing argument 1 of 'free' discards 'const' qualifier from pointer target type`。

**原因**：`src` 声明为 `static const char *src;`，而 `free` 收 `void *`。

**正确写法**：`free((void *)src);`——显式丢掉 const。
这在语义上有点别扭（`src` 指向的是 `malloc` 出来的可写内存，只是我们选择用 const 视图读它），
所以这个转换是安全的。

### 坑 10：`unary()` 里用 `match()` 导致 token 丢失

**现象**：写

```c
if (match(T_MINUS)) {
    Node *n = node(N_UNARY, *cur());   /* ← 错！cur() 已经前进了 */
```

结果 `n->op` 拿到的是**下一个** token 的类型。

**正确写法**：先 `Token t = *cur();` 再 `pos++`，
或者用 `expect()`（但 `expect` 遇到非 `-` 会报错，不适合这里）。

## 9. 自测题

### 9.1 概念题

**Q1.** 为什么需要 AST？不能"边解析边执行"吗？

<details>
<summary>答案</summary>

因为**运算符优先级**。要正确处理 `1 + 2 * 3`，必须先知道 `*` 比 `+` 优先。
"边解析边执行"的程序在解析到 `+` 时还不知道右边会不会冒出 `*`，来不及调整。

AST 把"形状"（语法）和"数值"（语义）分成两步：
第 2 步只管把形状建出来，第 3 步只管从叶子往回算。

第二个好处：**AST 可以打印给人看**（`ast` 模式就是靠这个），
第三个好处：同一棵 AST 以后可以换一个后端去执行。
</details>

**Q2.** `match()` 和 `expect()` 有什么区别？

<details>
<summary>答案</summary>

| | `match(t)` | `expect(t, what)` |
|---|---|---|
| 匹配成功 | `pos++`，返回 1 | `pos++`，返回那个 `Token` |
| 匹配失败 | 什么都不做，返回 0 | 报错 + `exit(1)` |
| 语义 | "这里**可能**有" | "这里**必须**有" |

典型例子：参数列表用 `match(T_COMMA)`（逗号可有可无），
用 `expect(T_RP, "函数调用需要 ')'")`（右括号必须）。
</details>

**Q3.** 为什么 `node()` 里必须 `memset`？

<details>
<summary>答案</summary>

因为 `nodes` 是**静态数组**，第 2 次及以后复用某个槽位时，
里面还留着上一次解析的数据。如果某个字段没被显式赋值就读取，
读到的就是**上一次的值**（不是随机的，但一样是错的）。

`memset` 保证每个字段在被显式赋值前都是确定的 0。

**一个实测过的例子**：去掉 `memset` 后，`N_CALL` 节点的 `count` 残留着上一个
调用的参数个数，于是 `g()`（零参数）会打印出上一个 `f(1,2,3)` 的 3 个参数。

**这个 bug 极其阴险**：单表达式测试全过，多表达式或循环里才炸，
而且症状出现在**和真正原因相隔很远的地方**。
</details>

**Q4.** 为什么 `bin_level` 的参数名 `next` 有点误导？

<details>
<summary>答案</summary>

`next` 指的是"**下一层**"，也就是**优先级更高**的那一层
（`bin_level(unary, ...)` 里的 `unary` 比 `unary` 自身更高）。
但"next"读起来像"下一个"，容易理解反。

叫 `higher` 或 `sub` 会更清楚。这是一个可以从本步学到的**命名要贴合语义**的例子。
</details>

### 9.2 预测题（改一行会发生什么）

**Q1.** 把 `add()` 里的 `bin_level(mul, o, 2)` 改成 `bin_level(unary, o, 2)`，会发生什么？

<details>
<summary>答案</summary>

`+` 和 `-` 的右操作数会直接用 `unary()` 解析，
也就是 `*` `/` `%` 不再比 `+` `-` 优先。

`1 + 2 * 3` 的树会变成：

```text
二元 PLUS
├── 数字 1
└── 二元 STAR
    ├── 数字 2
    └── 数字 3
```

等等——**这个例子恰好看不出差别**！因为 `mul()` 遇到 `*` 时会继续，
而 `unary()` 遇到 `*` 会直接返回（`*` 不是一元运算符）。

准确地说：`unary()` 解析 `2` 后返回 `数字 2`，
`add()` 看到下一个 token 是 `*`（不在 `{T_PLUS, T_MINUS}` 里），于是返回，
`expression()` 返回 `PLUS(1, 2)`，然后 `mode_ast` 检查到当前 token 是 `*` 不是 EOF，
报 `错误 [1:7]: 语法错误：表达式结束后还有多余的 '*'`。

**验证**：

```sh
printf '1 + 2 * 3\n' > /tmp/e.ml
./minilang /tmp/e.ml ast
```

（更明显的例子是 `1 * 2 + 3`，会变成 `STAR(1,2)` 然后报"多余的 '+'"。）
</details>

**Q2.** 把 `mode_ast()` 里的 `if (cur()->type != T_EOF)` 那个检查删掉，会发生什么？

<details>
<summary>答案</summary>

多余的 token 会被**静默丢弃**。

```sh
printf '1 2\n' > /tmp/x.ml
./minilang /tmp/x.ml ast
```

删之前：`错误 [1:3]: 语法错误：表达式结束后还有多余的 '2'`，退出码 1。
删之后：正常打印 `二元 PLUS`（其实是 `数字 1`）的树，退出码 0。

用户以为程序跑成功了，其实少算了一大半。
</details>

**Q3.** 把 `primary()` 里 `if (match(T_LP))` 那个分支删掉，会发生什么？

<details>
<summary>答案</summary>

括号表达式无法解析。`(1 + 2) * 3` 会报：

```text
错误 [1:1]: 语法错误：此处需要数字、变量、函数调用或括号表达式
```

因为 `(` 既不是数字也不是标识符，走到 `primary()` 最后那个 `fatal_at`。

**这说明括号不是"可选的语法糖"，它是 `primary()` 的四大分支之一。**
</details>

**Q4.** 把 `bin_level` 里的 `left = n;` 删掉，会发生什么？

<details>
<summary>答案</summary>

每次循环都从**最初的** `left` 出发，`1 + 2 + 3` 会变成：

```text
二元 PLUS
├── 数字 1
└── 二元 PLUS
    ├── 数字 2
    └── 数字 3
```

——**右结合**了！值从 `(1+2)+3 = 6` 变成 `1+(2+3) = 6`（加法看不出差别），
但减法立刻露馅：`1 - 2 - 3` 会从 -4 变成 2。

```sh
printf '1 - 2 - 3\n' > /tmp/e.ml
./minilang /tmp/e.ml ast
```

**`left = n` 这一行就是左结合的全部秘密。**
</details>

**Q5.** 把 `Node` 里的 `char text[MAX_TEXT]` 改成 `char text[16]`，会发生什么？

<details>
<summary>答案</summary>

**短变量名会丢字符。** 语法分析器用 `snprintf(n->text, sizeof(n->text), "%s", t.text)` 拷贝，
`sizeof(n->text)` 变成 16，所以超过 15 个字符的名字会被**静默截断**。

`snprintf` 不报错，只截断——这就是它在本步和第 1 步表现不同的原因
（第 1 步的词法分析器用 `memcpy` + 长度检查，会明确报"标识符太长"）。

**验证**：写一个 20 个字符的变量名，语法树里会显示成前 15 个字符。
</details>

### 9.3 动手题

**Q1.** 让 `ast` 模式能打印"表达式的节点总数"。

<details>
<summary>答案</summary>

在 `Node` 结构体里没法直接加计数（它是每个节点自己的），
最简单的办法是加一个全局计数器，在 `node()` 里加：

```c
static int nnodes;      /* 已经分配了多少个节点 */
```

`nnodes` 在 `node()` 里就是"已分配数"，所以在 `mode_ast` 里
`printf("共 %d 个节点\n", nnodes);` 就能打印。

注意 `nnodes` 是**全局**的，多次解析会累加。要每次重置，
在 `mode_ast` 里 `nnodes = 0;`（放在 `lex()` 之后、`expression()` 之前）。

验证：

```sh
printf '1 + 2 * 3\n' > /tmp/e.ml
./minilang /tmp/e.ml ast
```

应该显示 `共 5 个节点`（1 个 PLUS + 1 个 STAR + 3 个数字）。
</details>

**Q2.** 在 `dump_ast` 里给每层加上**括号**，让树形更明确。

<details>
<summary>答案</summary>

思路：在 `N_BINARY` 分支里，根据运算符优先级决定要不要加括号。
最简单的版本（所有二元节点都加）：

```c
case N_BINARY:
    printf("二元 %s\n", type_name((TokType)n->op));
    kids[m++] = n->a;
    kids[m++] = n->b;
    break;
```

改成：

```c
case N_BINARY:
    printf("(二元 %s)\n", type_name((TokType)n->op));
```

更精确的做法是按优先级判断——但这需要知道每个运算符的优先级，
而本步没有这张表的一个**可查询**版本（只有 `bin_level` 里的调用顺序）。

**这其实是个很好的观察**：优先级表目前是"隐式"的（藏在函数调用链里），
要显式查询就得把它抽成一张数据表——那会改动 `bin_level` 的设计，
超出本步范围。

验证：跑 `tests/case_01_precedence.ml`，根节点会变成 `(二元 PLUS)`。
</details>

**Q3.** 让 `ast` 模式支持**多个表达式**（用分号分隔），每个都打印一棵树。

<details>
<summary>答案</summary>

思路：把 `mode_ast` 里的"解析一个 + 检查 EOF"改成"循环"。

```c
static int mode_ast(const char *path) {
    src = read_file(path);
    lex();
    while (cur()->type != T_EOF) {
        Node *e = expression();
        printf("语法树：\n");
        dump_ast(e, "");
        if (!match(T_SEMI)) break;      /* 分号是可选的分隔符 */
    }
    free((void *)src);
    return had_error ? 1 : 0;
}
```

**注意**：`match(T_SEMI)` 是"试探"的，所以最后一个表达式后面没有分号也能工作。
但如果循环因为 `match(T_SEMI)` 失败而 `break`，说明还有多余的 token，
应该再报一次"多余的 token"错误。

验证：

```sh
printf '1 + 2; 3 * 4\n' > /tmp/m.ml
./minilang /tmp/m.ml ast
```

应该打印两棵树。
</details>

**Q4.**（进阶）让 `ast` 模式支持 `let x = 1 + 2;` 这样的语句。

<details>
<summary>答案</summary>

本步的 AST 节点类型里**没有** `N_LET`。你需要加：

1. `NodeType` 枚举加 `N_LET`；
2. `Node` 结构体加一个字段存变量名——可以复用 `text`，
   但要注意 `N_LET` 有两个孩子（名字和值），得用 `a` / `b`：
   `a` = `N_VAR` 节点（变量名），`b` = 值的表达式；
3. 写一个 `statement()` 函数：

```c
static Node *statement(void) {
    if (match(T_LET)) {
        Token t = *cur();
        Token name = expect(T_ID, "变量名");
        expect(T_ASSIGN, "变量定义需要 '='");
        Node *v = expression();
        Node *n = node(N_LET, t);
        n->a = node(N_VAR, name);
        n->b = v;
        return n;
    }
    return expression();
}
```

4. `dump_ast` 加 `case N_LET:` 分支；
5. `mode_ast` 改成调用 `statement()` 而不是 `expression()`。

**这个练习的价值**：它让你意识到"语句层"和"表达式层"是两套东西，
而且**语句需要新的节点类型**。第 5 步会正式做这件事。
</details>

## 10. 本步的设计取舍

**为什么先做 AST 而不是直接写解释器？**

因为**可验证性**。第 2 步的 `ast` 模式能让你**看见**程序被理解成了什么。
如果直接写求值器，`1 + 2 * 3` 算错了你根本不知道是优先级搞错了还是求值写错了。
有了 AST，你可以先把树打印出来、盯着看几遍、确认形状对了，再往下写求值。

这在教学里的价值特别大——**每一步都有一个可以"眼见为实"的产物**。

**为什么用 `bin_level` 这个通用循环，而不是每层写一个 `while`？**

| | `bin_level` 通用版 | 每层一个 `while` |
|---|---|---|
| 代码量 | 一处 20 行 + 每层 3 行 | 6 处 × 8 行 = 48 行 |
| 加一层优先级 | 加 3 行函数 | 加 8 行代码 |
| 优先级可见性 | **藏在调用链里**，要数 | 一眼可见，代码顺序即优先级 |
| 出 bug 的可能性 | 调错调用顺序 | 复制粘贴漏改 |

本步选了通用版，因为**加优先级是高频操作**（第 4 步就要加比较和逻辑）。
代价是优先级的顺序变得不明显——这正是你在 9.2 Q1 里被坑到的地方。

**一个折中：可以在 `bin_level` 旁边加注释，把优先级链明确写出来。**
本步的代码里每个函数上方的注释（`/* * / % */`、`/* + - */` …）就在做这件事。

**为什么节点池不用 `malloc`？**

见 5.3。核心是"AST 是一次性建好、然后只读"的，
释放它没有意义，而"不用管释放"能省掉一大堆 bug。

**还没做什么？**

- `ast` 模式只支持**一个表达式**，不支持语句、不支持分号分隔的多个表达式。
- 不支持 `let`、`if`、`for`、`fn` 的语法（第 5~10 步做）。
- 不支持一元运算符 `!` 的求值（语法能解析，第 3 步求值时才报错）。
- `Node.params[]` / `nparams` 已经预留但没用（第 10 步用）。
- `Node.c` 这个子节点指针已经预留但没用。
- 没有做**语义检查**（比如"变量是否已定义"）——那是第 5 步的事。

## 11. 下一步预告

现在你有了 AST：一棵纯粹的"形状"树，`1 + 2 * 3` 变成了
`PLUS(1, STAR(2, 3))`，但它还不会算。

下一步我们写**求值器（evaluator）**：一个 `eval()` 函数，
从树的**叶子**开始，一个节点一个节点地往上算。
你会看到一件很反直觉的事：

> **语法分析是"从左往右展开"，求值是"从叶子往回算"——两个方向的递归。**

你会学到：
- `Value` 结构体为什么现在就要预留 `is_string` 字段（第 9 步加字符串时才不用改签名）；
- 为什么 `eval` 要提前带一个 `Env *e` 参数（第 5 步才用得上，但那时就不用改所有调用点了）；
- C 的 `%` 运算符为什么不能用于 `double`，取余要用 `fmod()`；
- 除零检查应该放在哪里。

第 3 步的 `ast` 模式会多打印一行 `求值结果：7`。

→ [第 3 步 · AST 求值器（算术）](../003/README.md)
