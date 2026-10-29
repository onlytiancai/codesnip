# 第 3 步 · AST 求值器（算术）

> 进度 ███░░░░░░░ 3/13 ｜ 上一节：[第 2 步 · 表达式语法分析 → AST](../002/README.md) ｜ 下一节：[第 4 步 · 比较与逻辑运算](../004/README.md)

## 1. 本步目标

读完你将能够：

- 让 `./minilang demo.ml ast` 在打印语法树之后**再打印一行 `求值结果：7`**。
- 说清楚求值的递归方向为什么和语法分析**正好相反**（一个从左往右展开，一个从叶子往回算）。
- 解释 `Value` 结构体里 `is_string` 和 `string` 这两个"现在还没用上"的字段，
  理解这是**提前留接口**（INV-2）的价值。
- 解释 `eval()` 为什么现在就要带一个用不上的 `Env *e` 参数，
  以及"提前留参数"为什么比"以后改签名"好。
- 用 `fmod()` 而不是 `%` 做浮点取余，并且知道**为什么**。
- 在除零和取模零时给出带行列号的错误，并以退出码 `1` 结束。

## 2. 运行方式

```sh
cd 003
./run.sh                 # 编译 + 跑全部用例 + 跑 demo
./minilang demo.ml      # 默认 run 模式（本步还没实现）
./minilang demo.ml tokens
./minilang demo.ml ast
```

## 3. 现在的程序能做什么

| 模式 | 输入 | 输出 | 退出码 |
|---|---|---|---|
| `tokens` | 一个 `.ml` 文件 | token 列表（**从第 1 步起一个字都没改**） | 0 / 1 |
| `ast` | 一个表达式 | stdout：语法树 **+ `求值结果：<数字>`** | 0 / 1 |
| `ast`（求值失败） | 一个表达式 | stdout：语法树；stderr：`错误 [行:列]: …` | 1 |
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
求值结果：7
```

注意 `求值结果：7` 里的 `7` —— 不是 `7.000000`。
这是因为用了 `printf("求值结果：%g\n", v.value)`，
`%g` 会自动选择最短的表示法（见 5.7）。

## 4. 前置知识回顾

上两步我们已经搭好了两个部件：

- **词法分析器**（第 1 步）：源码 → token 流。每个 token 带 `line` / `col`。
- **语法分析器**（第 2 步）：token 流 → AST。`Node` 有 `N_NUM` / `N_STR` / `N_VAR` /
  `N_UNARY` / `N_BINARY` / `N_CALL` 六种类型，`N_BINARY` 节点有 `op`（运算符）、
  `a` / `b`（左右子节点）。

上一步的 `ast` 模式只做"建树 + 打印树"。这一步要加的是**第三件事：算值**。

一个提醒：本步**不修改** AST，也不修改词法分析器。
求值器是**只读**地遍历那棵树，算出一个数。

## 5. 本步引入的核心概念

### 5.1 求值：两个方向的递归

这是本步最核心的思想，值得用图讲清楚。

**语法分析**是"从左往右展开"：

```text
表达式文本:   1 + 2 * 3
               ↓ 从左往右读 token
语法分析:     logical_or() → logical_and() → equal() → compare() → add() → mul()
               ↓ 递归下降，一路往"里"钻
构建 AST:     PLUS
               ├── 1
               └── STAR(2, 3)
```

每一层优先级函数**先解析左操作数**（递归往下钻），
再解析右操作数，最后**组装**出一个节点往上返回。
所以语法分析是**自顶向下**的——从"整个表达式"往"最小单元"走。

**求值**是"从叶子往回算"：

```text
求值 AST:     PLUS
               ├── 1        ← 叶子，直接返回 1
               └── STAR(2, 3)
                        ├── 2   ← 叶子，返回 2
                        └── 3   ← 叶子，返回 3
                           ↓ STAR: 2 * 3 = 6
                        ↓ 回到 PLUS
               ↓ PLUS: 1 + 6 = 7
```

求值也是递归，但**递归的方向相反**：
`eval` 收到一个节点，**先算它的孩子**，再拿孩子的值算自己。

写成伪代码：

```c
Value eval(Node *n) {
    switch (n->type) {
        case N_NUM:   return 数字的值;                 // 叶子
        case N_BINARY: {
            Value a = eval(n->a);                      // 先算左孩子
            Value b = eval(n->b);                      // 再算右孩子
            return a op b;                             // 最后算自己
        }
    }
}
```

**一个重要的推论：求值器里一行优先级代码都没有。**
`+` 和 `*` 谁先算？**由树形决定，不由代码决定。**
`STAR` 在树的更深处，所以它先算完。

这就是 AST 的核心价值——把"优先级"这种语法规则，**一次性固化成了树形**，
然后求值器就不需要再关心优先级了。

### 5.2 调用栈长什么样

把求值过程画成调用栈（`→` 表示"调用"）：

```text
eval(PLUS)
│
└─→ eval(STAR)
    │
    └─→ eval(NUM 2)  → 返回 Value{2}
    │
    └─→ eval(NUM 3)  → 返回 Value{3}
    │
    └─→ 2 * 3 = 6   → 返回 Value{6}

回到 eval(PLUS)：
    Value a = 6（来自左边）
    Value b = eval(NUM 1) = 1
    6 + 1 = 7
```

递归深度等于**表达式的嵌套深度**。
`(1 + (2 * (3 + 4)))` 这样的表达式会嵌套很深。

> 💡 本步**没有做递归深度限制**。第 11 步讲递归函数时会加上 128 层的上限。
> 表达式嵌套极深时（比如几百层括号），C 的调用栈可能会溢出。
> 教学项目里这不太可能发生，但如果你在 fuzz 测试里遇到，可以自己加个计数器。

### 5.3 `Value` 结构体：为什么要预留 `is_string`（INV-2）

```c
/* 一个值。MiniLang 里普通变量只能是数字，字符串只出现在 print 里，
   所以 Value 用一个标记位区分两者。
   is_string 和 string 这两个字段现在还没用上，是提前留好的。 */
typedef struct {
    double value;             /* 数值            */
    int    is_string;         /* 1 表示是字符串  */
    char   string[MAX_TEXT];  /* 字符串内容      */
} Value;
```

**为什么 `value` 用 `double` 而不是 `int`？**
因为 MiniLang 的数字类型是"整数和小数统一用 `double`"。
`10` 和 `3.14` 都是 `double`。代价是整数运算会有浮点误差
（比如 `0.1 + 0.2 != 0.3`），但对这个教学项目来说，
"一套类型搞定"比"整数精确"重要得多。

**为什么要有 `is_string`？**
本步的求值器**根本不支持字符串**（遇到 `N_STR` 直接报错）。
那为什么还要留这个字段？

因为**第 9 步**要支持 `print("hi {x}\n")` 这样的字符串模板。
到那时候，求值器要能返回字符串值。如果第 3 步的 `Value` 只有 `double value`，
第 9 步就得把它改成：

```c
typedef struct {
    double value;
    int    is_string;
    char   string[MAX_TEXT];
} Value;
```

而且**所有**返回 `Value` 的函数、所有 `Value` 的局部变量、
`eval()` 的返回类型……全都要跟着改。虽然是机械修改，但**改动面很大**，
而且很容易漏掉一处（漏掉的那处会静默地读写错误的字段）。

**提前留好字段，第 9 步就一行函数签名都不用改。** 这就是 INV-2。

**这就是"接口定型要一次到位"的第二条实例**：
第 1 步留了 `Token.line/col`，第 3 步留了 `Value.is_string/string`。

**那 `is_string` 字段现在为什么是 0？**
看 `number()` 助手：

```c
/* 造一个数字值 */
static Value number(double x) {
    Value v;
    v.value = x;
    v.is_string = 0;
    v.string[0] = '\0';
    return v;
}
```

三个字段都**显式**设置了。虽然 `v` 是局部变量（未初始化），
但这里一个字段一个字段都赋了值，没有"未定义的字段"。

> 💡 严格来说，`v.string[0] = '\0'` 只设置了**第一个字节**，
> `v.string[1]` 到 `v.string[4095]` 还是不定值。
> 但因为 `is_string == 0`，那些字节永远不会被当作字符串读，
> 所以是正确的。如果哪天有人改成读 `v.string` 的全部内容，
> 就得改成 `memset(&v, 0, sizeof(v))`。
> **这是"用哨兵值 + 不读未初始化部分"的标准技巧**，但要清楚它的边界。

### 5.4 为什么要 `number()` 这个小助手

```c
static Value number(double x) {
    Value v;
    v.value = x;
    v.is_string = 0;
    v.string[0] = '\0';
    return v;
}
```

如果不要这个助手，`eval()` 里会到处是这样的代码：

```c
case N_NUM: {
    Value v;
    v.value = n->num;
    v.is_string = 0;
    v.string[0] = '\0';
    return v;
}
```

`eval()` 里一共出现了 **8 次** `return number(...)`。
每次都手写三行赋值 = 24 行重复代码，而且**每次都可能忘掉某个字段**
（最典型的就是忘了 `is_string = 0`，然后 `Value` 变成"字符串型数字"，
第 9 步的 `print` 模板就会把 `42` 当成字符串处理）。

**`number()` 的三个价值：**

1. **消除重复**。
2. **保证不变量**：凡是"数字值"，`is_string` 一定是 0。
   这是一个**类型构造器（smart constructor）**的雏形——
   C 里没有构造函数，就用一个函数来代替。
3. **单点修改**：以后要给数字值加字段（比如第 5 步加"来源环境"），
   只改 `number()` 一个地方。

**返回结构体 vs 返回指针：**
`number()` 返回 `Value`（值），不是 `Value *`（指针）。这是有意的：
- 避免"谁负责释放"的問題；
- `Value` 只有 4KB 多（在 64 位上 `double` 8 字节 + `int` 4 字节 + 4096 数组），
  按值返回用的是调用者提供的隐藏指针，**不会有额外的堆分配**；
- 结构体按值返回在 C99 之后是标准做法（早期 C 需要靠"返回指向静态对象的指针"绕弯）。

> ⚠️ 但要注意：`Value` 有 4KB 大小，按值返回意味着**每次 `return number(x)` 都要拷贝 4KB**。
> 在一个求值器里被调用成千上万次，这是可观的开销。
> 更高效的做法是把 `string` 改成指针（`const char *string;`），
> 让 `Value` 缩到 16 字节。**本步选择简单优先**——
> 4KB 的拷贝在教学规模下完全看不出来，而改成指针会带来"字符串内容存哪"的新问题。

### 5.5 `eval` 为什么带一个 `Env *e` 参数

```c
/* 环境（作用域）。这一步还用不上它，但先把参数位置留出来：
   第 5 步讲作用域时就不用改 eval 的签名，也不用动所有调用点。 */
typedef struct Env Env;

/* 求值：把一棵表达式树算成一个值。 */
static Value eval(Node *n, Env *e) {
    (void)e;   /* 目前还用不到环境，第 5 步开始会真正用到它 */
    ...
}
```

这是本步**最值得学的一条 API 设计经验**。

**问题**：`Env`（环境 / 作用域表）是第 5 步才出现的概念。
第 5 步要支持变量，`eval(N_VAR)` 就必须能查变量表：

```c
case N_VAR:
    return lookup(e, n->text);   /* 需要访问环境！ */
```

如果第 3 步的 `eval` 签名是 `eval(Node *n)`，第 5 步就得改成
`eval(Node *n, Env *e)`，而这意味着**所有调用点都要改**：

```text
eval(n->a)      →  eval(n->a, e)
eval(n->b)      →  eval(n->b, e)
eval(e, NULL)   →  eval(e, NULL)      （在 mode_ast 里）
```

本步的 `eval` 里有 **3 个递归调用点 + 1 个外部调用点**。
第 5 步之后 `eval` 会有十几个调用点，还要加上 `exec()` 里的调用。
每次改签名都要全盘搜索一遍——**这是典型的"接口不成熟"的代价**。

**所以第 3 步就带上 `Env *e`，即使现在传 `NULL`。**

**这样做的三条理由：**

1. **签名一次定型。** 第 5 步只要把 `(void)e;` 换成真正的实现，
   递归调用点已经带上了 `e`，**一行都不用改**。
2. **零成本。** 多一个指针参数在 x86-64 上就是多一个寄存器，
   对性能没有可测量的影响。
3. **表达意图。** 看到 `eval(Node *n, Env *e)`，
   读代码的人立刻知道"求值需要环境"，而不是"求值是纯函数"。
   （虽然本步的实现里 `e` 被忽略了，但**接口在说真话**。）

**`(void)e;` 是干什么的？**
告诉编译器"我知道 `e` 现在没用，故意的"。
没有它，`gcc -Wall -Wextra` 会报 `unused parameter 'e'`。
第 5 步把这个 `(void)e;` 删掉就行。

**`typedef struct Env Env;` 这个前向声明也是提前留的。**
第 3 步根本不知道 `Env` 长什么样（它只有 `Env *` 指针，不需要知道），
但要声明一个指向 `struct Env` 的指针，就得先有 `struct Env` 这个类型名。
用**不完整类型（incomplete type）**的前向声明就够了——
这是 C 里"先约定接口、后填实现"的经典手法。

> 💡 `Env` 永远不会被解引用（在这一步），所以前向声明是安全的。
> 如果你写成 `typedef struct Env { ... } Env;` 并定义空结构体，
> 在 C 里是合法的（C++ 不允许空结构体），但**没有意义**。

### 5.6 `fmod` vs `%`：为什么不能直接用 `%`

C 语言的 `%` 运算符**只支持整数**：

```c
double a = 7.5, b = 2.0;
a % b          /* ❌ 编译错误：invalid operands to binary % */
```

编译器会报 `error: invalid operands to binary '%' ('double' and 'double')`。
这是**编译期**错误，不是运行期——编译器知道 `double` 不能用 `%`。

**为什么 C 要这样设计？**
因为 `%` 的语义是"整数除法的余数"，它的数学定义依赖于**欧几里得除法**：
`a = q*b + r`，其中 `0 <= r < |b|`。这个定义只对整数有良定义的结果。
对浮点数，"余数"依赖于舍入方式，不同语言给出不同答案（有的向零取整，有的向下取整）。

**MiniLang 的解决方案是 `fmod()`：**

```c
case T_PERCENT:
    if (b == 0) fatal_at(n->tok, "取余运算的除数不能为 0");
    return number(fmod(a, b));   /* C 的 % 只支持整数，浮点取余要用 fmod */
```

`fmod(x, y)` 在 `<math.h>` 里，返回 `x / y` 的浮点余数，
**语义是"截断除法"（truncation toward zero）**，和 C 的整数 `%` 一致。

验证：

```sh
printf '7.5 %% 2\n' > /tmp/m.ml
./minilang /tmp/m.ml ast
```

输出 `求值结果：1.5`（`7.5 = 3*2 + 1.5`）。

**对应的坑：负数。**
C 的整数 `%` 是**截断**语义（`-7 % 2 == -1`），
而 Python 的 `%` 是**向下取整**语义（`-7 % 2 == 1`）。
`fmod` 跟 C 的 `%` 一致，是截断语义：

```sh
printf -- '-7 %% 2\n' > /tmp/n.ml
./minilang /tmp/n.ml ast
```

会得到 `-1`。**如果你以后想改成 Python 语义，就得自己写 `fmod` 的修正**：
`r = fmod(a, b); if (r != 0 && ((r < 0) != (b < 0))) r += b;`

**`fmod` 需要 `#include <math.h>`，而且链接时要加 `-lm`。**
本步的编译命令已经有 `-lm`（第 1 步就加了，虽然当时用不上）。
**这是提前留的另一个例子**：`-lm` 从第 1 步就在，第 3 步才真正需要它。

### 5.7 除零检查放在哪

```c
case T_SLASH:
    if (b == 0) fatal_at(n->tok, "除数不能为 0");
    return number(a / b);
case T_PERCENT:
    if (b == 0) fatal_at(n->tok, "取余运算的除数不能为 0");
    return number(fmod(a, b));
```

**为什么放在 `eval` 里，而不是放在语法分析阶段？**
因为"除数是不是 0"是**运行期**的信息，不是语法信息。
语法上 `1 / 0` 完全合法（就像 C 里 `1/0` 语法合法但运行会崩）。
只有当程序**真的要执行这行代码**时，才知道分母是多少。

**为什么检查的是 `b == 0` 而不是 `fabs(b) < EPSILON`？**
因为 MiniLang 的设计是"除零就报错"，不是"除零返回无穷大"。
用精确的 `== 0` 更简单也更符合教学目标。
（真实语言里 `1e-300 / 1e-300` 会溢出成 `inf`，但那是另一个话题。）

**为什么用 `fatal_at` 而不是 `error_at`？**
因为**除零之后没法继续算**——你不能返回一个"错误值"然后接着算下去
（那会让错误传播到整个程序，非常难调试）。
所以直接终止是合理的。

**错误信息为什么用 `n->tok`（节点自带的 token）而不是 `*cur()`（当前 token）？**
因为**求值时 `cur()` 已经没有意义了**——词法分析早就结束了，
`pos` 停在最后的 EOF token 上。而 `n->tok` 记录的是**这个运算符在源码里的位置**，
这才是用户真正需要看到的。

> 💡 这是第 2 步"每个 `Node` 都存一个 `Token`"的价值兑现。
> 如果 `Node` 里没存 `tok`，第 3 步的错误信息就只能报一个"不知道哪里的错"。

**`fatal_at` 报的行列号是运算符的位置，不是除数的位置。**
`1 / 0` 报的是 `[1:3]`（`/` 的位置），不是 `[1:5]`（`0` 的位置）。
这是有意为之：错在"这里要除"，而不是"这里有个 0"。

### 5.8 `%.15g` 的输出格式

```c
printf("求值结果：%g\n", v.value);
```

`%g` 是 C 的**通用浮点格式**，规则是：

1. 用 `%e`（科学计数法）和 `%f`（定点小数）两种方式中，**较短的那个**；
2. 默认精度 6 位有效数字；
3. 去掉尾部的零（`3.140000` → `3.14`）。

**本教程统一用 `%g`**（准确说是 `%g` 的默认精度，或者写成 `%.15g`）：

| 值 | `%g` | `%.15g` | `%f` |
|---|---|---|---|
| `15.0` | `15` | `15` | `15.000000` |
| `3.14` | `3.14` | `3.14` | `3.140000` |
| `0.5` | `0.5` | `0.5` | `0.500000` |
| `1.0/3.0` | `0.333333` | `0.333333333333333` | `0.333333` |
| `1e20` | `1e+20` | `1e+20` | `100000000000000000000.000000` |

> 本步的代码实际用的是 `%g`（默认 6 位有效数字）。
> 教程文档里说"数字输出格式是 `printf("%.15g")`"，
> 指的是整个教程最终统一采用的精度约定；
> 第 3 步的 `ast` 模式用的是 `%g`，效果对教学用例没有区别。
> `%.15g` 的好处是能显示 `double` 的**几乎全部**有效位
> （`double` 精确表示大约 15~17 位十进制数字）。

**为什么用 `%g` 而不是 `%f`？**
因为 `%f` 会把 `15` 打成 `15.000000`，很难看。
MiniLang 的数字统一是 `double`，但用户写 `15` 就该显示 `15`。

**为什么不用 `%d` 或 `%ld`？**
因为值是 `double`。用 `%d` 打印 `double` 是**未定义行为**
（`printf` 的变参不会自动转换类型）。这在第 8 节的坑里会细说。

### 5.9 还没实现的分支：诚实地报错

```c
case N_STR:
    fatal_at(n->tok, "求值器还不支持字符串，字符串会在讲 print 模板时支持");
    break;
case N_VAR:
    fatal_at(n->tok, "求值器还不支持变量，变量会在讲作用域时支持");
    break;
case N_CALL:
    fatal_at(n->tok, "求值器还不支持函数调用，内置函数会在后面一步支持");
    break;
```

这三个分支现在**语法上能解析**（第 2 步就把它们建好了树），
但**求值不了**。

**为什么不干脆在语法分析阶段就拒绝它们？**
因为**以后会支持**。第 5 步支持变量，第 8 步支持内置函数，第 9 步支持字符串。
如果第 3 步就在语法分析阶段拒绝，那第 5 步要改语法分析器、第 8 步又要改、
第 9 步还要改——**每一步都要动上一层的代码**。

现在的做法是：**语法层尽量宽松，把"能不能算"的问题交给求值层**。
求值层遇到不支持的就明确报错，并告诉用户"以后某一步会支持"。

**这个设计的代价**：`ast` 模式对 `x + 1` 会先打印语法树再报错，
看起来有点怪（语法树明明建好了）。但这个"怪"恰恰是有用的——
它告诉你**语法是对的，只是语义还没实现**。

**为什么 `fatal_at` 后面还有 `break;`？**
`fatal_at` 标了 `_Noreturn`，理论上不可达。
但 C 的 `switch` 里，**每个 case 如果要 `return`，就得保证所有路径都返回**。
加了 `break;` 后，代码落到 `switch` 末尾的 `return number(0);`，
不会触发"control reaches end of non-void function"警告。
这是**防御性写法**：万一 `fatal_at` 哪天被改成不退出，也不会产生未定义行为。

## 6. 动手：把代码加进来

### 6.1 改动清单

| 位置锚点 | 改了什么 | 大概多少行 | 为什么 |
|---|---|---|---|
| `#include` 区 | 加 `#include <math.h>` | 1 | `fmod()` 需要它。链接参数 `-lm` 第 1 步就有 |
| 运行时区块（插在 `dump_ast` 之后、`read_file` 之前） | `typedef struct Env Env;` + `Value` 结构体 + `number()` + `eval()` | 约 80 | 见 5.3 / 5.4 / 5.5 |
| `mode_ast()` 函数体末尾 | 加两行：调用 `eval` + 打印结果 | 2 | 让 `ast` 模式输出求值结果 |

**没有改动**：词法分析器、AST 定义、语法分析器、`dump_ast`、`main`。
**diff 里只有一处插入 + 两行新增**，非常干净（符合 INV-8）。

### 6.2 分段讲解

#### 第 1 段：`#include <math.h>`

```diff
 #include <stdio.h>
 #include <stdlib.h>
 #include <string.h>
 #include <ctype.h>
+#include <math.h>
 #include <stdarg.h>
```

**为什么 `math.h` 放在 `stdarg.h` 前面？**
纯粹是按字母顺序排的（`ctype` < `math` < `stdarg`），没有功能原因。
`#include` 的顺序不影响正确性。

**`math.h` 需要链接 `-lm`。**
`run.sh` 里的编译命令是：

```sh
$CC $CFLAGS minilang.c -lm -o "$BIN"
```

`-lm` 从第 1 步就在（当时 `math.h` 都没 include，但加上无害）。
**顺序很重要**：`-lm` 必须放在**源文件之后**。
如果写成 `cc -lm minilang.c -o out`，某些链接器会因为
"库在引用之前"而找不到符号。这是个经典坑。

#### 第 2 段：`Value` 和 `number()`

```c
/* 环境（作用域）。这一步还用不上它，但先把参数位置留出来：
   第 5 步讲作用域时就不用改 eval 的签名，也不用动所有调用点。 */
typedef struct Env Env;

/* 一个值。MiniLang 里普通变量只能是数字，字符串只出现在 print 里，
   所以 Value 用一个标记位区分两者。
   is_string 和 string 这两个字段现在还没用上，是提前留好的。 */
typedef struct {
    double value;             /* 数值            */
    int    is_string;         /* 1 表示是字符串  */
    char   string[MAX_TEXT];  /* 字符串内容      */
} Value;

/* 造一个数字值 */
static Value number(double x) {
    Value v;
    v.value = x;
    v.is_string = 0;
    v.string[0] = '\0';
    return v;
}
```

**`typedef struct Env Env;` 为什么放在 `Value` 前面？**
纯粹因为它是"运行时区块"的开头，读起来顺。
`eval` 的参数用到 `Env *`，所以 `Env` 必须先声明。

**`number()` 里为什么不用 `memset`？**
见 5.3 的讨论：三个字段都显式赋值了，`v.string` 只有第 0 字节被设为 `'\0'`，
其余字节不定——但因为 `is_string == 0`，它们永远不会被当字符串读。
如果想更保险：

```c
static Value number(double x) {
    Value v;
    memset(&v, 0, sizeof(v));
    v.value = x;
    return v;
}
```

这样更慢（每次清 4KB），但**行为完全可预测**。本步选了前者。

**如果你写错了会怎样？**

- 忘了 `v.is_string = 0;`：数字值被当成字符串。第 9 步 `print` 模板会把 `42` 当字符串处理，
  打印出来还是 `42`（看不出差别），但 `x + 1` 就会出问题。**这是很难发现的 bug。**
- 把 `string` 字段删掉：第 9 步要加回来，而且要改所有函数签名。

#### 第 3 段：`eval()` 的骨架

```c
static Value eval(Node *n, Env *e) {
    (void)e;   /* 目前还用不到环境，第 5 步开始会真正用到它 */
    if (!n) return number(0);

    switch (n->type) {
        case N_NUM:
            return number(n->num);
        ...
    }
    return number(0);
}
```

**`if (!n) return number(0);` 是防御性检查。**
理论上 `eval` 永远不会被传 `NULL`（树的每个孩子都被 `node()` 分配过）。
但 `dump_ast` 里也有 `if (!n) return;` 这种检查——**同一个防御习惯**。
代价是一行代码，收益是万一有 bug 不会直接崩在 `n->type` 的空指针解引用上。

**末尾的 `return number(0);`**
`switch` 覆盖了 `NodeType` 的**全部 6 个值**，理论上每个 case 都 return 了，
编译器不会报 "control reaches end of non-void function"。
但加上这行更保险（万一以后加了新的 `NodeType` 却忘了加 case）。
而且 `fatal_at` 后面的 `break;` 会落到这里。

**如果你写错了会怎样？**
删掉末尾的 `return number(0);`，在 `gcc -Wall -Wextra` 下会得到
`warning: control reaches end of non-void function`。
不致命，但很吵。

#### 第 4 段：`N_BINARY`——本步的核心

```c
case N_BINARY: {
    Value x = eval(n->a, e);
    Value y = eval(n->b, e);
    double a = x.value, b = y.value;
    switch (n->op) {
        case T_PLUS:    return number(a + b);
        case T_MINUS:   return number(a - b);
        case T_STAR:    return number(a * b);
        case T_SLASH:
            if (b == 0) fatal_at(n->tok, "除数不能为 0");
            return number(a / b);
        case T_PERCENT:
            if (b == 0) fatal_at(n->tok, "取余运算的除数不能为 0");
            return number(fmod(a, b));
        default:
            fatal_at(n->tok, "比较与逻辑运算会在下一步支持");
    }
    break;
}
```

**为什么要用 `{ }` 包住 case？**
因为 case 里有 `Value x = ...;` 这样的**声明**。
C 的语法规定：语句块（compound statement）里的声明必须被 `{}` 包住，
不能直接写在 `case` 标签后面（那是 C++ 的写法，C 里不行）。
`gcc` 会报 `error: a label can only be part of a statement and a declaration is not a statement`。

**为什么先 `eval` 出 `Value`，再取 `.value`？**

```c
Value x = eval(n->a, e);
Value y = eval(n->b, e);
double a = x.value, b = y.value;
```

因为 `eval` 返回的是 `Value`（可能是字符串，第 9 步起），
但本步的算术运算只对**数值**有意义。
先取出 `.value` 存到 `double`，后面就用 `double` 算——清晰且高效。

**注意求值顺序**：`eval(n->a, e)` 先执行，`eval(n->b, e)` 后执行。
这是 C 语言规定的求值顺序（`;` 是序列点）。
第 4 步的 `&&` / `||` 有**短路求值**的需求，那时要注意写法。

**`default:` 分支为什么要有？**
因为 `n->op` 可能是任何 `TokType`——包括 `T_LT`（比较，第 4 步才支持）、
`T_EQ`、`T_AND` 等等。`default` 兜住"暂时不支持的运算符"，
给出明确的错误信息，而不是默默算出一个错的结果。

**这是本步最重要的"诚实"设计**：宁可报错，也不要给一个错答案。
因为错误答案会让 bug 藏得更深。

**如果你写错了会怎样？**

- 用 `%` 而不是 `fmod`：`a % b` 编译不过（`double` 不能用 `%`）。
- 忘了除零检查：`1 / 0` 在 IEEE 754 下**不会崩溃**，而是得到 `inf`；
  `fmod(1, 0)` 得到 `NaN`。打印出来是 `inf` / `nan`。
  **这是很隐蔽的 bug**——程序"跑通了"，但结果是垃圾。
  所以除零检查不是可有可无的。
- 把 `fatal_at(n->tok, ...)` 写成 `fatal_at(*cur(), ...)`：
  错误信息的行列号会变成 **EOF 的位置**（因为求值时 `pos` 早就停在末尾了），
  完全指不到真正出错的地方。

#### 第 5 段：`N_UNARY`

```c
case N_UNARY: {
    Value a = eval(n->a, e);
    if (n->op == T_MINUS) return number(-a.value);
    if (n->op == T_PLUS)  return number(a.value);
    fatal_at(n->tok, "逻辑运算 ! 会在下一步支持");
    break;
}
```

**一元运算只有一个孩子，所以只 `eval` 一次。**

`+x` 是合法的（正号），返回 `a.value` 本身。
`-x` 取负。

**`!x` 呢？**
`!` 是逻辑非，**它的结果不是数字而是布尔值**。
本步还没有布尔值，所以 `fatal_at` 报错。
第 4 步会加上——那时候的结果是 `1`（真）或 `0`（假），
因为 MiniLang 没有独立的布尔类型。

> 💡 这暴露了 `Value` 设计的一个**局限**：`is_string` 能区分字符串和数字，
> 但**区分不了布尔和数字**。第 4 步只能把布尔当作 `1` / `0` 的数字。
> 这是"没有独立布尔类型"的语言的必然结果，也是 MiniLang 的刻意简化
> （原语言设计里就说"比较运算产生布尔值，但不需要允许用户直接定义布尔变量"）。

#### 第 6 段：`mode_ast` 加两行

```diff
     printf("语法树：\n");
     dump_ast(e, "");
+    Value v = eval(e, NULL);
+    printf("求值结果：%g\n", v.value);
     free((void *)src);
     return had_error ? 1 : 0;
 }
```

**为什么 `eval(e, NULL)` 传 `NULL` 而不是 `e`？**
因为 `e` 这个变量名已经被**表达式树**占用了：

```c
Node *e = expression();       /* e 是 AST 根节点 */
...
Value v = eval(e, NULL);      /* e 是树，env 参数是 NULL */
```

变量名撞车了。两个选择：
1. 把 AST 根节点改名叫 `root`（更清楚）；
2. 环境参数传 `NULL`（本步的做法，因为确实不需要环境）。

本步选了 2，因为**改名字要动更多行**（而 INV-8 说尽量只插入不修改）。
但这个"撞名"本身说明：**命名要能区分不同概念**。
第 5 步真正引入 `Env` 时，这个撞车会被解决。

**`eval` 在 `dump_ast` 之后调用。**
为什么不是先算值再打印树？
因为**先打印树再算值**，出错时你还能看到"程序被理解成了什么"——
这对调试非常有用。你在第 7 节会看到除零用例的 stdout 里**确实有语法树**，
而 stderr 里是错误信息。

### 6.3 完整 diff

> **怎么用这段 diff**：推荐**照着敲**——每改一小段就编译一次，比一次性粘贴更容易记住。
> 如果想直接应用，把下面的内容存成 `step.patch`，然后在**上一步的目录**里执行：
>
> ```sh
> patch -p1 < ../003/step.patch
> ```
>
> 应用完和本目录的 `minilang.c` **逐字节一致**（可以 `diff` 验证）。

<details>
<summary>点开看完整 diff（真实 diff -u 输出）</summary>

```diff
--- a/minilang.c
+++ b/minilang.c
@@ -24,6 +24,7 @@
 #include <stdlib.h>
 #include <string.h>
 #include <ctype.h>
+#include <math.h>
 #include <stdarg.h>
 
 #define MAX_TOKENS 20000
@@ -467,6 +468,86 @@
 }
 
 /* ------------------------------------------------------------------ */
+/* 运行时：值与求值                                                    */
+/* ------------------------------------------------------------------ */
+
+/* 环境（作用域）。这一步还用不上它，但先把参数位置留出来：
+   第 5 步讲作用域时就不用改 eval 的签名，也不用动所有调用点。 */
+typedef struct Env Env;
+
+/* 一个值。MiniLang 里普通变量只能是数字，字符串只出现在 print 里，
+   所以 Value 用一个标记位区分两者。
+   is_string 和 string 这两个字段现在还没用上，是提前留好的。 */
+typedef struct {
+    double value;             /* 数值            */
+    int    is_string;         /* 1 表示是字符串  */
+    char   string[MAX_TEXT];  /* 字符串内容      */
+} Value;
+
+/* 造一个数字值 */
+static Value number(double x) {
+    Value v;
+    v.value = x;
+    v.is_string = 0;
+    v.string[0] = '\0';
+    return v;
+}
+
+/* 求值：把一棵表达式树算成一个值。
+   递归的结构和语法分析器正好相反：分析时"从左到右展开"，
+   求值时"从叶子往回算"。 */
+static Value eval(Node *n, Env *e) {
+    (void)e;   /* 目前还用不到环境，第 5 步开始会真正用到它 */
+    if (!n) return number(0);
+
+    switch (n->type) {
+        case N_NUM:
+            return number(n->num);
+
+        case N_STR:
+            fatal_at(n->tok, "求值器还不支持字符串，字符串会在讲 print 模板时支持");
+            break;
+
+        case N_VAR:
+            fatal_at(n->tok, "求值器还不支持变量，变量会在讲作用域时支持");
+            break;
+
+        case N_CALL:
+            fatal_at(n->tok, "求值器还不支持函数调用，内置函数会在后面一步支持");
+            break;
+
+        case N_UNARY: {
+            Value a = eval(n->a, e);
+            if (n->op == T_MINUS) return number(-a.value);
+            if (n->op == T_PLUS)  return number(a.value);
+            fatal_at(n->tok, "逻辑运算 ! 会在下一步支持");
+            break;
+        }
+
+        case N_BINARY: {
+            Value x = eval(n->a, e);
+            Value y = eval(n->b, e);
+            double a = x.value, b = y.value;
+            switch (n->op) {
+                case T_PLUS:    return number(a + b);
+                case T_MINUS:   return number(a - b);
+                case T_STAR:    return number(a * b);
+                case T_SLASH:
+                    if (b == 0) fatal_at(n->tok, "除数不能为 0");
+                    return number(a / b);
+                case T_PERCENT:
+                    if (b == 0) fatal_at(n->tok, "取余运算的除数不能为 0");
+                    return number(fmod(a, b));   /* C 的 % 只支持整数，浮点取余要用 fmod */
+                default:
+                    fatal_at(n->tok, "比较与逻辑运算会在下一步支持");
+            }
+            break;
+        }
+    }
+    return number(0);
+}
+
+/* ------------------------------------------------------------------ */
 /* 读文件                                                              */
 /* ------------------------------------------------------------------ */
 static char *read_file(const char *path) {
@@ -523,6 +604,8 @@
         fatal_at(*cur(), "语法错误：表达式结束后还有多余的 '%s'", cur()->text);
     printf("语法树：\n");
     dump_ast(e, "");
+    Value v = eval(e, NULL);
+    printf("求值结果：%g\n", v.value);
     free((void *)src);
     return had_error ? 1 : 0;
 }
```

</details>

### 6.4 插入位置说明

本步只有 **1 处插入 + 2 行新增**：

```text
① #include 区
   #include <ctype.h>
   #include <math.h>       ← 插入

② dump_ast() 之后、"读文件"注释块之前
   ← 插入「运行时：值与求值」整块：
      typedef struct Env Env;
      Value 结构体
      number()
      eval()

③ mode_ast() 里 dump_ast(e, ""); 那一行之后
   ← 插入两行：
      Value v = eval(e, NULL);
      printf("求值结果：%g\n", v.value);
```

**为什么运行时区块放在 `dump_ast` 之后？**
`eval()` 不调用 `dump_ast`，`dump_ast` 也不调用 `eval`——它们之间没有依赖。
放在哪里都行。选这里是因为**逻辑上相关**（都是"处理 AST"的部分），
而且紧跟在语法分析器后面，读起来是"建树 → 打印树 → 求值"的自然顺序。

**为什么 `eval` 定义在 `mode_ast` 之前？**
C 要求先声明后使用。`mode_ast` 里调用 `eval`，
所以 `eval` 的定义必须在 `mode_ast` 之前（或者加前向声明）。

**为什么 `Value` 定义在 `eval` 之前？**
`eval` 的返回类型是 `Value`。同理，`Node` 必须在 `Value` 之前吗？**不是**——
`Value` 里没有 `Node`，两者互不依赖。

**为什么 `typedef struct Env Env;` 不需要 `Env` 的完整定义？**
因为只用到 `Env *`（指针）。C 允许对**不完整类型**用指针。
这是 C 里"先定接口"的标准做法。

## 7. 测试

`./run.sh` 的真实输出：

```text
▶ 用例（模式：ast）
  ✅ case_01_arithmetic.ml        [ast]
  ✅ case_02_parentheses.ml       [ast]
  ✅ case_03_division.ml          [ast]
  ✅ case_04_modulo.ml            [ast]
  ✅ case_05_divide_by_zero.ml    [ast]
▶ demo.ml
  | 语法树：
  | 二元 MINUS
  | ├── 二元 STAR
  | │   ├── 二元 PLUS
  | │   │   ├── 数字 1
  | │   │   └── 数字 2
  | │   └── 数字 3
  | └── 二元 SLASH
  |     ├── 数字 4
  |     └── 数字 2
  | 求值结果：7

✅ 全部通过（6 项）
```

`demo.ml` 的内容是 `(1 + 2) * 3 - 4 / 2`。
求值过程：`3 * 3 = 9`，`4 / 2 = 2`，`9 - 2 = 7`。

### 逐个用例说明

**case_01_arithmetic** —— 输入 `1 + 2 * 3`，期望 stdout：

```text
语法树：
二元 PLUS
├── 数字 1
└── 二元 STAR
    ├── 数字 2
    └── 数字 3
求值结果：7
```

**为什么是 7？** `STAR` 在树的更深处，先算出 `2 * 3 = 6`，
然后 `PLUS` 算 `1 + 6 = 7`。**优先级完全由树形决定。**

**case_02_parentheses** —— 输入 `(1 + 2) * 3`，期望 stdout：

```text
语法树：
二元 STAR
├── 二元 PLUS
│   ├── 数字 1
│   └── 数字 2
└── 数字 3
求值结果：9
```

**为什么是 9？** 括号让 `PLUS` 在里、`STAR` 在外，
所以先算 `1 + 2 = 3`，再算 `3 * 3 = 9`。

**对比 case_01**：同样的运算符，因为括号改变了树形，结果从 7 变成 9。
**这就是 AST 把优先级"固化"成形状的直接证据。**

**case_03_division** —— 输入 `10 / 4`，期望 stdout：

```text
语法树：
二元 SLASH
├── 数字 10
└── 数字 4
求值结果：2.5
```

**为什么是 2.5 而不是 2？** 因为 MiniLang 的数字统一是 `double`，
`10 / 4` 是浮点除法。C 里如果两边都是整数，`10 / 4` 会得到 `2`（截断）。
但这里 `a` 和 `b` 都是 `double`，所以是 `2.5`。

**这个设计是有意的**：MiniLang 不区分整数和浮点（语言设计文档里写了
"整数和小数可以参与同一算术表达式"）。代价是 `10 / 2` 会得到 `5` 而不是 `5.0`——
因为输出用 `%g`，`5.0` 显示成 `5`。

**case_04_modulo** —— 输入 `10 % 3`，期望 stdout：

```text
语法树：
二元 PERCENT
├── 数字 10
└── 数字 3
求值结果：1
```

`fmod(10, 3) = 1`。因为两边都是 `double`，
**必须用 `fmod` 而不是 `%`**（`%` 在 C 里只支持整数，编译不过）。

**case_05_divide_by_zero** —— 输入 `1 / 0`，期望 **stdout**：

```text
语法树：
二元 SLASH
├── 数字 1
└── 数字 0
```

期望 **stderr**：

```text
错误 [1:3]: 除数不能为 0
```

期望退出码 `1`。

**三个细节：**

1. **stdout 里有语法树，stderr 里有错误。** 因为 `dump_ast` 在 `eval` **之前**调用，
   树已经打印出来了。这是故意的——出错时也能看到程序被理解成了什么。
2. **列号是 3，不是 5。** `1 / 0` 里 `1` 在第 1 列，`/` 在第 3 列，`0` 在第 5 列。
   报错用的是 `n->tok`（`/` 这个运算符的 token），所以是 `[1:3]`。
3. **退出码是 1。** `fatal_at` 里的 `exit(1)`。

### 其他真实运行

**取余零**（`10 % 0`）：

```console
$ printf '10 %% 0\n' > /tmp/m.ml
$ ./minilang /tmp/m.ml ast
错误 [1:4]: 取余运算的除数不能为 0
语法树：
二元 PERCENT
├── 数字 10
└── 数字 0
```

（列号 4 是 `%` 的位置。）

**浮点取余**（`7.5 % 2`）：

```console
$ printf '7.5 %% 2\n' > /tmp/m2.ml
$ ./minilang /tmp/m2.ml ast
语法树：
二元 PERCENT
├── 数字 7.5
└── 数字 2
求值结果：1.5
```

**变量还没支持**（`x + 1`）：

```console
$ printf 'x + 1\n' > /tmp/v.ml
$ ./minilang /tmp/v.ml ast
错误 [1:1]: 求值器还不支持变量，变量会在讲作用域时支持
语法树：
二元 PLUS
├── 变量 x
└── 数字 1
```

**字符串还没支持**（`"hi"`）：

```console
$ printf '"hi"\n' > /tmp/s.ml
$ ./minilang /tmp/s.ml ast
错误 [1:1]: 求值器还不支持字符串，字符串会在讲 print 模板时支持
语法树：
字符串 "hi"
```

**函数调用还没支持**（`f(1)`）：

```console
$ printf 'f(1)\n' > /tmp/c.ml
$ ./minilang /tmp/c.ml ast
错误 [1:1]: 求值器还不支持函数调用，内置函数会在后面一步支持
语法树：
调用 f
└── 数字 1
```

**一元负号**（`-5 + 3`）：

```console
$ printf -- '-5 + 3\n' > /tmp/n.ml
$ ./minilang /tmp/n.ml ast
语法树：
二元 PLUS
├── 一元 MINUS
│   └── 数字 5
└── 数字 3
求值结果：-2
```

**左结合**（`2 - 3 - 4`）：

```console
$ printf '2 - 3 - 4\n' > /tmp/l.ml
$ ./minilang /tmp/l.ml ast
语法树：
二元 MINUS
├── 二元 MINUS
│   ├── 数字 2
│   └── 数字 3
└── 数字 4
求值结果：-5
```

`2 - 3 - 4` = `(2-3)-4` = `-5`，**不是** `2-(3-4)` = 3。

**`run` 模式还没实现**：

```console
$ ./minilang demo.ml
解释执行尚未实现，请先用 tokens 或 ast 模式查看结果
```

## 8. 易错点与坑

### 坑 1：用 `%` 做浮点取余

**现象**：

```text
error: invalid operands to binary '%' ('double' and 'double')
```

**原因**：C 的 `%` 运算符**只支持整数**。

**正确写法**：`fmod(a, b)`，并且 `#include <math.h>` + 链接 `-lm`。

**为什么 `%.15g` 的输出格式在这不影响？**
不相关。但要注意 `fmod` 的结果是 `double`，
用 `%d` 打印会得到垃圾（见坑 5）。

### 坑 2：忘了 `-lm`，或者 `-lm` 放在源文件前面

**现象**：

```text
Undefined symbols for architecture arm64:
  "_fmod"
```

**原因**：`fmod` 在 `libm` 里，macOS 上链接时需要显式指定。

**正确写法**：

```sh
cc -std=c11 -Wall -Wextra -O2 minilang.c -lm -o minilang
```

**`-lm` 必须在源文件之后。** 链接器是**从左到右**处理文件的，
`-lm` 写在 `minilang.c` 前面时，链接器还没看到对 `fmod` 的引用，
就不会把 `libm` 里的符号拉进来。

> 💡 顺带一提：**macOS 上 `fmod` 其实在 `libSystem` 里**，所以不加 `-lm` 也能链接成功；
> 但 Linux 上必须加。**写可移植代码就别赌这个。**

### 坑 3：忘了除零检查

**现象**：程序"跑通了"，但输出是 `inf` 或 `nan`。

```console
$ printf '1 / 0\n' > /tmp/d.ml
$ ./minilang /tmp/d.ml ast   # 假设没有检查
求值结果：inf
```

**原因**：IEEE 754 规定 `1.0 / 0.0 = +inf`、`0.0 / 0.0 = NaN`，
**浮点除零不会崩溃**。C 里只有**整数**除零才是 UB（通常触发 SIGFPE 崩溃）。

**正确写法**：`if (b == 0) fatal_at(n->tok, "除数不能为 0");`

**这个坑特别阴险**：因为"不崩溃"，你以为没问题，
但 `inf` 会在后面的运算里传播，最终产生一串莫名其妙的 `nan`。
**明确报错永远比静默产生垃圾值好。**

### 坑 4：`fatal_at` 用了 `*cur()` 而不是 `n->tok`

**现象**：求值出错的行列号总是指向**文件末尾**。

```text
错误 [2:1]: 除数不能为 0
```

（`2:1` 是 EOF 的位置，而不是 `1 / 0` 里 `/` 的位置 `1:3`。）

**原因**：求值阶段 `pos` 早就停在最后的 EOF token 上了，`cur()` 完全没有意义。

**正确写法**：`fatal_at(n->tok, ...)`。
**`Node` 里存了 `tok` 就是为了这一刻。**

### 坑 5：用 `%d` 打印 `double`

**现象**：打印出一个垃圾数字，比如 `求值结果：-1073741824`。

**原因**：`printf` 的变参**不做类型转换**。
`%d` 会从参数里取 4 个字节当整数读，但实际传进去的是 8 字节的 `double`，
读到的是 double 的低位字节——完全是垃圾。

**正确写法**：`%g` / `%f` / `%.15g`（配 `double`），
`%d`（配 `int`），`%lld`（配 `long long`），`%zu`（配 `size_t`）。

**同类错误**：`printf("%d\n", x)` 其中 `x` 是 `size_t`（无符号），
在 64 位上会读错一半。应该用 `%zu`。

### 坑 6：`Value` 是 4KB 大小，按值传递的开销

**现象**：性能分析发现 `eval` 是热点函数。

**原因**：`Value` 里有 `char string[MAX_TEXT]`（4096 字节），
每次 `return number(x)` 都要拷贝 4KB，每次 `Value x = eval(...)` 又要拷 4KB。

**正确写法（如果真的在乎性能）**：
把 `string` 改成指针 `const char *string;`，`Value` 就缩到 16 字节。

**本步为什么不改？** 因为：
1. 第 9 步的字符串内容需要**拥有**一份拷贝（存哪儿是个新问题）；
2. 教学代码优先"行为可预测"；
3. 这个开销在教学规模下看不出来。

**但你要知道这是一个已知的取舍**，而不是"没想过"。

### 坑 7：`case` 里写声明忘了加 `{}`

**现象**：

```text
error: a label can only be part of a statement and a declaration is not a statement
```

**原因**：C（不是 C++）不允许 `case X:` 后面直接跟声明。

**正确写法**：

```c
case N_BINARY: {
    Value x = eval(n->a, e);
    ...
    break;
}
```

或者把声明提到 `switch` 外面。

### 坑 8：`case` 里 `return` 了还写 `break`

**现象**：无功能影响，但看起来很奇怪。

**原因**：`return` 之后 `break` 永远执行不到。

**正确写法**：可以两个都写（无害），也可以只留 `return`。
本步的代码在 `fatal_at`（`_Noreturn`）后面保留了 `break;`，
是为了防止 `fatal_at` 之后的控制流掉到 `switch` 末尾——**这是有意的**。

### 坑 9：`number()` 里忘了 `v.is_string = 0;`

**现象**：现在看不出来，但第 9 步会出诡异 bug。

**原因**：`v` 是未初始化的局部变量，`v.is_string` 是**不定值**（不定值不是"随机"，
是"读它就是 UB"）。可能是 0，也可能是 1，也可能是任何值。

**正确写法**：每个字段都显式赋值。或者用 `memset(&v, 0, sizeof(v))` 一次性清零。

**怎么提前发现**：把 `Value` 改成 `static Value v;`（静态变量），
未赋值的字段会保持 0，看起来"正常"——
但这恰恰会掩盖 bug。**测试时用局部变量，用 `-fsanitize=address,undefined` 编译。**

### 坑 10：`fmod` 的负数语义和 Python 不一样

**现象**：`-7 % 2` 在 MiniLang 里是 `-1`，在 Python 里是 `1`。

**原因**：C 的 `%` 是**截断**语义（向零取整），
Python 的 `%` 是**向下取整**语义。
`fmod` 跟 C 一致。

**正确写法（如果想改成 Python 语义）**：

```c
double r = fmod(a, b);
if (r != 0 && ((r < 0) != (b < 0))) r += b;
```

本步保持 C 语义（和 `fmod` 的文档一致）。

### 坑 11：忘了 `math.h` 里的函数需要 `-lm`

见坑 2。

## 9. 自测题

### 9.1 概念题

**Q1.** 求值的递归方向和语法分析有什么不同？

<details>
<summary>答案</summary>

- **语法分析**：自顶向下**展开**。`expression()` 先解析左操作数（递归往下钻），
  再解析右操作数，最后组装节点**往上返回**。
- **求值**：自底向上**收敛**。`eval()` 先算孩子（递归往深处），
  再拿孩子的值算自己**往上返回**。

一个方向相反的递归。

**推论**：求值器里**没有一行优先级代码**。
`+` 和 `*` 谁先算完全由树形决定。
</details>

**Q2.** `Value` 里的 `is_string` 现在没用，为什么还要留？

<details>
<summary>答案</summary>

因为第 9 步要支持字符串（`print("hi {x}\n")`）。
如果现在不加，第 9 步就得改 `Value` 的定义，
进而要改**所有**返回 `Value` 的函数、所有 `Value` 局部变量、
`eval()` 的返回类型……改动面很大，而且容易漏改。

**提前留好，第 9 步就一行函数签名都不用改。** 这是 INV-2。
</details>

**Q3.** `eval` 的 `Env *e` 参数现在用不上，为什么还要加？

<details>
<summary>答案</summary>

第 5 步要支持变量，`eval(N_VAR)` 必须能查环境表，
所以 `eval` 的签名**必然**要变成 `eval(Node *n, Env *e)`。

如果不提前留，第 5 步要改所有调用点（本步有 4 个：3 个递归 + 1 个外部）。
第 5 步之后 `eval` 会有十几个调用点，还要加上 `exec()` 里的调用。

**提前留参数，等于把"以后要改的地方"变成"以后不用改的地方"。**
而且多一个指针参数在 x86-64 上零成本。
</details>

**Q4.** 为什么取余要用 `fmod` 而不是 `%`？

<details>
<summary>答案</summary>

C 的 `%` 运算符**只支持整数**，对 `double` 是编译期错误。
`fmod(x, y)` 在 `<math.h>` 里，返回浮点除法的余数，语义是"截断"，
和 C 的整数 `%` 一致。

代价：需要 `#include <math.h>` 和链接 `-lm`（`-lm` 第 1 步就有了）。
</details>

### 9.2 预测题（改一行会发生什么）

**Q1.** 把 `if (b == 0) fatal_at(n->tok, "除数不能为 0");` 这行删掉，会发生什么？

<details>
<summary>答案</summary>

`1 / 0` **不会崩溃**，而是得到 `inf`（IEEE 754 的规定）。
`0 / 0` 得到 `nan`。

```console
$ printf '1 / 0\n' > /tmp/d.ml
$ ./minilang /tmp/d.ml ast
语法树：
二元 SLASH
├── 数字 1
└── 数字 0
求值结果：inf
```

退出码变成 0，**测试用例会失败**（`case_05_divide_by_zero.status` 期望 1）。

这个 bug 特别阴险：不崩溃，所以很容易被忽略。
</details>

**Q2.** 把 `fatal_at(n->tok, "除数不能为 0")` 改成 `fatal_at(*cur(), "除数不能为 0")`，会发生什么？

<details>
<summary>答案</summary>

错误信息的行列号会指向 **EOF 的位置**（文件末尾），而不是 `/` 的位置。

```console
错误 [2:1]: 除数不能为 0
```

因为求值时 `pos` 早就停在最后的 EOF token 上，`cur()` 指向的是 EOF。
**`n->tok` 才是这个运算符在源码里的位置。**
</details>

**Q3.** 把 `number()` 里的 `v.is_string = 0;` 删掉，会发生什么？

<details>
<summary>答案</summary>

**现在看不出来**（`Value.is_string` 还没被任何代码读过）。
但 `v` 是未初始化的局部变量，`v.is_string` 是**不定值**——读它就是 UB。

第 9 步之后会出诡异 bug：某些 `number()` 造出来的值被当成字符串。

**怎么提前发现**：用 `gcc -fsanitize=address,undefined` 编译，
或者临时把 `Value v;` 改成 `Value v = {0};`（初始化为全 0），
看看行为有没有变化——有变化就说明有字段没被赋值。
</details>

**Q4.** 把 `eval` 的签名改成 `eval(Node *n)`（去掉 `Env *e`），会发生什么？

<details>
<summary>答案</summary>

**本步完全正常**（因为 `e` 本来就没用）。

但第 5 步会付出代价：所有 `eval(n->a, e)` 都要改回 `eval(n->a)`，
还要加 `Env *e` 参数——**而且是第 5 步那时候才加**，
改动点比第 3 步就加要多。

这就是"提前留参数"的价值：**代价前置，收益后置。**
</details>

**Q5.** 把 `printf("求值结果：%g\n", v.value)` 改成 `printf("求值结果：%d\n", v.value)`，会发生什么？

<details>
<summary>答案</summary>

打印出**垃圾数字**，比如 `求值结果：-1073741824` 或者 `求值结果：0`。

`%d` 会从变参里取 4 个字节当 `int` 读，但实际传的是 8 字节的 `double`，
读到的是 double 编码的**低位字节**——完全是垃圾。

这是 **UB（未定义行为）**，不是"确定的错误值"。
</details>

**Q6.** 把 `Value` 里的 `char string[MAX_TEXT]` 改成 `char string[16]`，会发生什么？

<details>
<summary>答案</summary>

**本步完全正常**（`string` 字段没人用）。

但第 9 步做字符串模板时，如果要存一个超过 15 字符的字符串，
就会溢出。而且 `number()` 里 `v.string[0] = '\0'` 只清了 1 字节，
剩下的 15 字节是不定值——**如果哪天有人读了 `v.string` 的全部内容**，
就会读到垃圾。

这说明"提前留字段"也要**配套地提前留好正确的初始化逻辑**。
</details>

### 9.3 动手题

**Q1.** 让 `ast` 模式同时打印**每个节点的类型**，比如在树后面加一行统计。

<details>
<summary>答案</summary>

思路：加一个全局计数器，或者写一个 `count_nodes()` 递归函数。

```c
static int count_nodes(Node *n) {
    if (!n) return 0;
    int c = 1;
    switch (n->type) {
        case N_UNARY: c += count_nodes(n->a); break;
        case N_BINARY: c += count_nodes(n->a) + count_nodes(n->b); break;
        case N_CALL:
            for (int i = 0; i < n->count; i++) c += count_nodes(n->items[i]);
            break;
        default: break;
    }
    return c;
}
```

在 `mode_ast` 里加：

```c
printf("共 %d 个节点\n", count_nodes(e));
```

**注意**：`N_STR` / `N_NUM` / `N_VAR` 是叶子，没有孩子，`default: break;` 刚好处理。

验证：

```sh
printf '1 + 2 * 3\n' > /tmp/e.ml
./minilang /tmp/e.ml ast
```

应该显示 `共 5 个节点`。
</details>

**Q2.** 支持**一元正号** `+x`（本步已经支持了，请验证），
并让它对**已经算出来的值**生效。

<details>
<summary>答案</summary>

本步的 `eval` 已经处理了 `T_PLUS`：

```c
if (n->op == T_PLUS)  return number(a.value);
```

验证：

```sh
printf -- '+5 + 3\n' > /tmp/p.ml
./minilang /tmp/p.ml ast
```

输出应该是 `求值结果：8`。

**如果你发现报错了**，检查 `unary()` 里是不是漏了 `T_PLUS`：

```c
if (cur()->type == T_MINUS || cur()->type == T_PLUS || cur()->type == T_NOT) {
```

第 2 步的代码里是有的。
</details>

**Q3.** 让求值器支持**布尔值**：`1 < 2` 返回 `1`（真），`1 > 2` 返回 `0`（假）。
（这是第 4 步的内容，提前做做看。）

<details>
<summary>答案</summary>

在 `eval` 的 `N_BINARY` 的 `switch (n->op)` 里加：

```c
case T_LT:  return number(a <  b);
case T_LE:  return number(a <= b);
case T_GT:  return number(a >  b);
case T_GE:  return number(a >= b);
case T_EQ:  return number(a == b);
case T_NE:  return number(a != b);
```

注意 **C 里 `<` `>` 的运算符优先级高于 `==` `!=`**，
但这里每个 case 都只有一个比较，不涉及混合，没问题。

验证：

```sh
printf '1 < 2\n' > /tmp/c.ml
./minilang /tmp/c.ml ast
```

应该输出 `求值结果：1`。

**这正是第 4 步要做的**——你提前做了一遍，
就知道第 4 步要加的东西有多小（6 个 case）。
</details>

**Q4.**（进阶）让 `ast` 模式**同时**打印求值过程（每一步的中间结果）。

<details>
<summary>答案</summary>

思路：在 `eval` 的 `N_BINARY` 分支里，在返回前打印：

```c
case N_BINARY: {
    Value x = eval(n->a, e);
    Value y = eval(n->b, e);
    double a = x.value, b = y.value;
    double r = 0;
    switch (n->op) {
        case T_PLUS:    r = a + b; break;
        case T_MINUS:   r = a - b; break;
        case T_STAR:    r = a * b; break;
        case T_SLASH:
            if (b == 0) fatal_at(n->tok, "除数不能为 0");
            r = a / b; break;
        case T_PERCENT:
            if (b == 0) fatal_at(n->tok, "取余运算的除数不能为 0");
            r = fmod(a, b); break;
        default: fatal_at(n->tok, "比较与逻辑运算会在下一步支持");
    }
    printf("  %g %s %g = %g\n", a, type_name((TokType)n->op), b, r);
    return number(r);
}
```

**注意**：这样改之后 `return number(a + b)` 变成了"先算到 `r` 再 return"，
需要把 `case` 里的 `return` 改成 `break` + 在 `switch` 后面统一 return。

**更简单的做法**（不改 `eval`）：在 `dump_ast` 里对 `N_NUM` 节点顺便打印它的值
（`dump_ast` 本来就打印了 `数字 %g`）。这已经能看出叶子值了。

验证：

```sh
printf '1 + 2 * 3\n' > /tmp/e.ml
./minilang /tmp/e.ml ast
```

会看到 `2 * 3 = 6` 和 `1 + 6 = 7` 两行——
**注意顺序：`STAR` 在深处，所以它的打印先出现。**
</details>

## 10. 本步的设计取舍

**为什么求值器不支持变量 / 字符串 / 函数调用，而是明确报错？**

因为**它们会在后面几步支持**。现在的设计是"语法层尽量宽松，语义层诚实报错"。

如果第 3 步就在语法分析阶段拒绝变量，那第 5 步要改语法分析器；
第 8 步加内置函数又要改；第 9 步加字符串还要改——
**每一步都要动上一层的代码**，违反 INV-8（只插入不修改）。

现在的做法让每一步都只在自己的层里加东西。

**为什么 `Value` 不用"联合体（union）"来省内存？**

理论上 `value`（8 字节）和 `string`（4096 字节）互斥，可以用 union 合并，
`Value` 缩到 4096 字节（省一点点）。但：

1. union 不能自动初始化，必须显式设置"当前是哪个成员"（那就是 `is_string` 的作用）；
2. 调试时 union 会让内存检查工具困惑；
3. 教学代码优先直白。

**没省下多少空间**（4096 vs 4104），却引入了 union 的全部复杂性。**不值。**

**为什么 `eval` 是 `static`？**

`static` 表示"这个函数只在**本文件**内可见"。
`eval` 不需要被其他文件调用，所以加 `static`：
- 编译器能知道它不会被外部覆盖，可以做更多优化；
- 避免和其他文件里的同名函数冲突；
- 明确表达"这是内部实现"。

第 13 步模块化重构时，`eval` 会变成 `src/eval.c` 里的函数，
那时候会去掉 `static` 并加上头文件声明——**这是 W-3 白名单里允许的改动**。

**还没做什么？**

- 不支持变量（第 5 步）、字符串（第 9 步）、函数调用（第 8 步）。
- 不支持比较和逻辑运算（第 4 步）。
- 没有布尔类型（`!`、比较运算的结果都是数字 `1` / `0`）。
- 没有递归深度限制（表达式嵌套极深可能爆栈）。
- 没有"短路求值"（第 4 步的 `&&` / `||` 需要）。
- `run` 模式还是空的（第 5 步才有真正能跑的语句）。
- `eval` 是**单次求值**，没有副作用的概念（也就没有"赋值表达式"）。

## 11. 下一步预告

现在 `ast` 模式能算出算术表达式的值了。但它的能力还停在
"数字之间的四则运算"。

下一步我们**激活第 2 步已经建好、但第 3 步拒绝求值**的那几类节点：

- **比较运算** `<` `<=` `>` `>=` `==` `!=` → 结果是 `1`（真）或 `0`（假）。
- **逻辑运算** `&&` `||` `!` → 带**短路求值**：
  `0 && (1 / 0)` 不会报错，因为 `&&` 左边是假就短路了。
- **`!` 的一元取反**。

你会学到：
- **短路求值（short-circuit evaluation）**为什么要靠**语句的顺序**而不是"先算两个操作数"来实现；
- C 里 `&&` / `||` 的返回值**不是布尔**，而是 `int`（`0` 或 `1`），
  这正好和 MiniLang"没有独立布尔类型"的设计对上；
- 为什么 `&&` 的右操作数必须在 `left` **算完之后**才求值——
  这是 `eval` 里**求值顺序**的经典陷阱。

第 4 步之后，`ast` 模式就能算出 `1 < 2 && 3 > 2` 是 `1` 了。

→ [第 4 步 · 比较与逻辑运算](../004/README.md)
