# C 版本 — branchless × NEON × stnp × nozero

父目录 Rust 版本（`../src/main.rs`）的 C 移植，同样 9 个实现 + memcpy 上限，同样的
SplitMix64 种子（`0xDEADBEEF_CAFEBABE`），同样的 min-over-30-iters benchmark。

只依赖 stdlib + `arm_neon.h` + inline asm。

## 怎么跑

```bash
make run          # 或者 make && ./branchless_filter
```

跑 2-3 次取稳定值——首次跑有冷缓存抖动，跟 Rust 版本一样。

## 稳定态结果（Apple M4，3 次运行取中位）

50% kept：

```
  variant          time      vs iter   vs copy_ceil
  filter_iter        3.27 ms    1.00x    21.8x
  prealloc           3.23 ms    1.01x    21.5x
  branchless         0.30 ms   10.9x      2.0x
  branchless_nozero  0.23 ms   14.2x      1.5x
  neon               0.31 ms   10.5x      2.1x
  neon_v2            0.27 ms   12.1x      1.8x
  neon_v2_nozero     0.23 ms   14.3x      1.5x
  neon_4lane         0.24 ms   13.6x      1.6x
  stnp (NT pair)     0.32 ms   10.0x      2.1x
  copy (ceiling)     0.15 ms
```

跟 Rust 版本对照，结论全部复现：

| | C | Rust |
|---|---:|---:|
| `filter_iter` | 3.27 ms | 3.10 ms |
| `branchless` | 0.30 ms | 0.29 ms |
| `branchless_nozero` | **0.23 ms** | **0.23 ms** |
| `neon_v2` | 0.27 ms | 0.30 ms |
| `neon_v2_nozero` | 0.23 ms | 0.26 ms |
| `stnp` | 0.32 ms | 0.38 ms |
| `copy_ceiling` | 0.15 ms | 0.17 ms |

- branchless 相对 iter 的 ~11x 提速 ✅
- 跳过零填充再省 ~20% ✅
- 手工 SIMD 没有显著胜过标量 branchless ✅
- `stnp` 依然是负优化 ✅

C 和 Rust 落在同一个区间内，说明这些结论是**微架构层面的**，跟语言/编译器前端无关——
两边都是 LLVM 后端，生成的循环基本一致。

## 移植时踩的两个坑

这两个坑都会**悄悄给出好看但错误的数字**，值得单独记一笔。

### 1. Dead store elimination 把整个循环删了

第一版跑出来一半的变体是 `0.00 ms`，summary 里全是 `infx` / `nanx`。

原因不是计时精度，是 benchmark 循环长这样：

```c
size_t len = f(input, n, threshold, &buf);
free(buf);                 // buf 从来没被读过
```

所有 filter 都是同一个 TU 里的 `static` 函数，clang 内联之后发现：这块内存分配出来、
写满、然后直接 `free` 掉，**没有任何人读它** → 整个写循环是 dead store → 删掉。

存活下来的恰好是编译器看不穿副作用的那几个，这个规律当时就是诊断线索：

| 变体 | 为什么没被删 |
|---|---|
| `filter_iter` | 循环里有 `realloc`，不透明副作用 |
| `*_nozero`（旧版） | 结尾 `realloc` 必须保留原内容 |
| `stnp` | inline asm 带 `"memory"` clobber |
| 其它全部 | ❌ 被删光，报 0.00 ms |

修法是加一个 `black_box`——Rust 那边 `std::hint::black_box` 干的同一件事：

```c
static inline void black_box(const void *p) {
    __asm__ __volatile__("" : : "r"(p) : "memory");
}
```

空 asm 把指针塞进寄存器并声明 clobber 掉 memory，编译器只能假设这里会读这块缓冲区，
于是必须把之前所有 store 落地。**必须放在计时区间内**，否则 store 可以被下沉到收尾的
`now_ms()` 之后。

### 2. 收缩 `realloc` ≠ `Vec::truncate`

修好 DSE 之后，`_nozero` 变体反而比 `calloc` 版本**慢**（0.42 vs 0.28 ms），跟 Rust
结论正好相反。而且它的耗时随 selectivity 单调上升（0.23 → 0.26 → 0.42 → 0.47 → 0.53），
其它列都是平的。**开销正比于保留元素数**，直接指向收尾那一行。

原来我用了收缩 `realloc` 去模拟 Rust 的 `truncate`：

```c
double *shrunk = realloc(buf, cur * sizeof(double));   // ← 错
```

我当时注释里还写着"缩小不会搬数据"——**这在 macOS 上是错的**。缩小会把块挪到更小的
size class，然后 `memcpy` 搬走保留的前缀。省下来的 8 MB memset 被这次拷贝吃干净了。

Rust 的 `Vec::truncate` 只改长度字段，**一个字节都不搬**。所以正确的移植是干脆不 realloc，
保持满容量的分配、只返回 `cur`：

```c
*out = buf;    // 保持 n 个 double 的容量
return cur;    // 只报告保留了多少
```

`[cur, n)` 仍然是未初始化的，但长度报的是 `cur`，没人会读到它；`double` 没有析构函数，
所以这是 sound 的——和 Rust 的 `with_capacity` + `set_len` + `truncate` 完全对应。

改完之后 `_nozero` 立刻回到 0.23 ms，各列也重新变平。

## 与 Rust 版本的实现差异

| | Rust | C |
|---|---|---|
| 零初始化 | `vec![0.0; n]` | `calloc(n, 8)` |
| 跳过零初始化 | `Vec::with_capacity` + `set_len` | `malloc(n * 8)` |
| 收尾截断 | `truncate(n)` | 直接返回 `cur`（不 realloc） |
| 防优化 | `std::hint::black_box` | 空 asm + `"memory"` clobber |
| `stnp` | `core::arch::asm!` | `__asm__ __volatile__` |
| 计时 | `Instant::now()` | `mach_absolute_time()` |

`stnp` 的 inline asm 两边形状一样，都要先把 `q` 寄存器拆成 2 个 `x` 寄存器（asm 不接受
SIMD 寄存器做 operand），这 2 条多出来的指令正是它慢 25% 的主因之一。

## 计时说明

macOS 上 `clock_gettime(CLOCK_MONOTONIC)` 的分辨率不足以测亚毫秒区间，改用
`mach_absolute_time()` + `mach_timebase_info()` 换算。非 Apple 平台回退到 `clock_gettime`。

## 适用架构

NEON 路径只在 `__aarch64__` 下编译（`HAVE_NEON`），其它架构自动回退到标量 branchless，
`make` 依然能过。
