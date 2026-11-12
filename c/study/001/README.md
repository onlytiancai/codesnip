# stats —— 一组数值的描述统计

一个 C11 单文件、零外部依赖的命令行工具，计算 **最大值 / 最小值 / 平均值 / 中位数 / 任意百分位数**（p99、p90、p85 等）。

## 编译

```bash
make                    # 生成 ./stats
# 或者手动：
cc -std=c11 -O2 -Wall -Wextra -pedantic -o stats stats.c -lm
```

## 快速开始

```bash
./stats 1 2 3 4 5                       # 命令行直接给数
printf '100\n50\n22\n46\n73\n' | ./stats
./stats -f data.txt                     # 从文件读，支持 # 注释
seq 1 100 | ./stats -c                  # 两行 CSV，便于脚本处理
./stats -p 99.9,99.5,75,25 -f data.txt  # 自定义百分位
./stats -m nearest -f data.txt          # 切换最近秩口径
./stats -h                              # 完整帮助
```

`printf '100\n50\n22\n46\n73\n' | ./stats` 的输出：

```
count      5
min        22
max        100
mean       58.2
median     50
p85        83.8
p90        89.2
p95        94.6
p99        98.92
method     linear
```

`-c` 输出两行 CSV（表头 + 数值），表头随 `-p` 变化：

```
count,min,max,mean,median,p85,p90,p95,p99
5,22,100,58.2,50,83.8,89.2,94.6,98.92
```

## 输入格式

| 来源 | 说明 |
| --- | --- |
| 命令行参数 | 支持负数、小数、科学计数法，如 `./stats -3.5 1e2 7` |
| 标准输入 | 无数值参数且未用 `-f` 时自动读取，支持多行 |
| `-f, --file PATH` | 从文件读取；`-` 表示 stdin；可与命令行数值混用 |

- 分隔符：空白（空格 / 制表 / 换行）、逗号、分号
- `#` 之后为注释
- 数值必须是**有限十进制数**；`nan`、`inf`、溢出值和 `12abc` 这类尾随垃圾会报错退出
- 字面的 `\n` / `\t` / `\r` 也会被当作空白。这是因为 zsh 的 `echo -n '100\n50\n'` 会解释转义、bash 的不会；跨 shell 建议统一用 `printf`

## 百分位口径

不同工具算出来的 p99 经常对不上，这里显式提供两种，用 `-m` 切换：

| 方法 | 定义 | 等价实现 | 1..100 的 p85 |
| --- | --- | --- | --- |
| `linear`（默认） | `h=(n-1)*p/100`，相邻样本线性插值 | numpy `percentile` 默认、Excel `PERCENTILE.INC`、R type 7 | 85.15 |
| `nearest` | 排序后第 `ceil(p/100*n)` 个数（1 起算） | 最近秩法，Prometheus 等监控面板口径 | 85 |

中位数固定用标准定义（偶数个样本取中间两数平均），与 `linear` 的 p50 一致。

默认百分位列表为 `85,90,95,99`，用 `-p, --percentiles` 覆盖，可传小数、会自动去重并保持输入顺序。

## 数量上限

**默认最多接收 1,000,000 个数值，超过立即报错退出**（避免误把超大输入灌进内存）：

```bash
./stats -n 100 -f data.txt                                            # 把上限调成 100
awk 'BEGIN{for(i=1;i<=1001;i++) printf "%d\n",i}' | ./stats -n 1000 -c  # 第 1001 个报错
./stats -n 0 -f huge.txt -c                                           # 0 表示不限制
```

验证边界不必真的造 100 万数据，用 `-n 1000` 缩小规模即可（测试就是这么做的）。生成精确数量的样本建议用 `awk`：macOS 的 `seq` 在数值 ≥1000000 时会额外多输出一行、并把数字写成 `1e+06`。

`-n, --max N` 接受非负整数，可放在数值参数之前或之后；上限对命令行数值、文件、stdin 一视同仁。

## 测试

```bash
make test               # 编译并跑全部测试
./tests/run_tests.sh    # 同上，会先增量编译，可单独执行
./tests/test_stats      # 只跑核心函数单元测试（需先编译）
./tests/test_cli.sh     # 只跑 CLI 端到端测试（需先编译）
```

测试无需第三方框架，包含：数值解析、两种百分位、均值/中位数、转义归一化、百分位列表解析、动态数组与上限逻辑，以及 CLI 的输出比对、错误处理、上限边界。边界测试用 `-n 1000` 缩小规模，不生成 100 万数据；默认的 1000000 由单元测试里的常量断言和帮助文本共同保证，整套测试不再需要生成百万级数据。

## 目录结构

```
.
├── stats.c                      # 主程序（单文件自包含）
├── Makefile                     # all / test / clean
├── README.md
└── tests/
    ├── test_stats.c             # 核心函数单元测试（极简断言宏）
    ├── test_cli.sh              # CLI 黑盒测试
    ├── run_tests.sh             # 一键运行全部测试
    └── data/
        ├── sample.txt           # 样例输入（多行、逗号、注释）
        └── expected.txt         # 期望输出基线，用于 diff
```

## 实现备注

- 平均值：先按最大绝对值缩放，再做 Kahan 补偿求和；中位数先各自减半再相加；异号大数插值改用凸组合。这几处都是为了在 `long double` 与 `double` 同宽的平台（如 Apple Silicon）上避免 `DBL_MAX` 级别数据溢出成 `inf`。
- 单元测试通过 `#define main` + `#include "../stats.c"` 复用被测的 `static` 函数，因此生产代码无需为了可测性拆成多个文件。
- `stats` 与 `tests/test_stats` 是编译产物，不必纳入版本控制（`make clean` 可清除）。
