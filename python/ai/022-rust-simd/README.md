# Branchless Rust × NEON × streaming store × no-zero-init — filter benchmark on Apple M4

复现 Serhii Potapov 的 [Branchless Rust: Making a Filter 4x Faster by Removing an if](https://www.greyblake.com/blog/branchless-rust/)，把 branchless filter 用 **aarch64 NEON intrinsics** 写成五个版本，**外加两个 mem-allocation 实验**（`stnp` 非临时写 / `Vec::with_capacity` 跳过零填充）一起对比。

只用 `std`，不引入 criterion。`src/main.rs` 是全部代码 + benchmark + 正确性校验。

## 文件结构

```
022-rust-simd/
├── Cargo.toml
├── README.md
├── src/
│   └── main.rs          # 全部代码 ~440 行
└── c/                   # C 移植版，同样 9 个实现（见 c/README.md）
    ├── branchless_filter.c
    ├── Makefile
    └── README.md
```

## 怎么跑

```bash
cargo run --release
```

期望输出：3 张表 + Summary。连续跑几次取稳定态——首次跑有冷缓存抖动。

## 九种实现

| 函数 | 思路 | 关键特性 |
|---|---|---|
| `filter_iter` | `iter().copied().filter().collect()` | idiomatic baseline（带数据依赖分支） |
| `filter_prealloc` | `Vec::with_capacity` + `if` 分支 | 去掉扩容开销，保留分支 |
| `filter_branchless` | 总是写 `out[n]=x`，游标按 `(x>threshold) as usize` 推进 | 控制依赖→数据依赖 |
| **`filter_branchless_nozero`** | branchless + `Vec::with_capacity` + unsafe `set_len` | 跳过 8 MB 输出零初始化 |
| `filter_simd_neon` | 2-lane NEON，SIMD compare + scalar store | `vcgtq_f64` 产 mask，scalar 写 2 次 |
| `filter_simd_neon_v2` | 2-lane NEON，in-register compaction + 1×`vst1q_f64` | `vextq` swap + `vbslq` select |
| **`filter_simd_neon_v2_nozero`** | v2 + 跳过零初始化 | 在 v2 基础上省 memset |
| `filter_simd_neon_4lane` | 4-lane NEON（v2 unrolled 2×），2×`vst1q_f64` | 减半 loop overhead |
| `filter_simd_neon_stnp` | v2 但用 `stnp` 替代 `vst1q_f64`，加非临时写提示 | 通过 `core::arch::asm!` 直接发指令 |

外加 `copy_ceiling = Vec::copy_from_slice` 作为内存带宽上限。

## NEON 实现细节

### v2 — in-register compaction + 向量存储

```rust
let lo      = vld1q_f64(...);              // [a, b]
let mask    = vcgtq_f64(lo, vth);          // [0xFF.. 或 0]
let swapped = vextq_f64(lo, lo, 1);        // [b, a]
let compact = vbslq_f64(mask, lo, swapped);// lane 0 = first_kept, lane 1 = b 或 garbage
vst1q_f64(out_ptr.add(n), compact);       // 一次写 16 字节
let pc = vaddvq_u64(vshrq_n_u64(mask, 63)) as usize;
n += pc;
```

### stnp — 非临时 pair store

`stnp` 是 AArch64 的 "store non-temporal pair" 指令（x86 对应 `movntpd`），**暗示 CPU**："这 16 字节是流式数据、写完不会再读，请绕过 cache"。

Rust stdarch 没暴露 `stnp` 的 typed intrinsic，用 `core::arch::asm!` 直接发指令：

```rust
let lo_u64: u64 = vget_lane_u64(vreinterpret_u64_f64(vget_low_f64(compact)), 0);
let hi_u64: u64 = vget_lane_u64(vreinterpret_u64_f64(vget_high_f64(compact)), 0);
core::arch::asm!(
    "stnp {lo}, {hi}, [{base}]",
    lo = in(reg) lo_u64, hi = in(reg) hi_u64,
    base = in(reg) out_ptr.add(n),
    options(nostack, preserves_flags)
);
```

### nozero — 跳过 `vec![0.0; N]` 的 8 MB 零填充

```rust
let mut out: Vec<f64> = Vec::with_capacity(input.len());
unsafe { out.set_len(input.len()) };  // "claim" the uninitialized buffer
// branchless loop writes out[n] always, advances n
// final out.truncate(n) discards the unwritten tail
```

**为什么安全**：branchless 循环对每个 input 元素都写一次 `out[n]`，所以 `[0, n_final)` 全部被真实值覆盖。`truncate(n_final)` 让 Vec 长度变成 `n_final`，放弃的尾部 `[n_final, N)` 永远不会被读到，且 `f64` 没有 `Drop`，所以 `set_len(N)` 是 sound 的。

## 在 Apple M4 上的稳定态结果

> N = 1,000,000 个 f64 = 8 MB 输入 / 8 MB 输出；均匀分布 [0.0, 100.0)；固定种子 `0xDEADBE…`；每个数据点 30 iters min（warmup=3），跑 4 次取后 3 次稳定值。**M4 有 4 MB L2 + ~16 MB 系统缓存**，8 MB 输入 + 8 MB 输出 = 16 MB 总量，**正好压在 system cache 边界**。

### Table 1 — selectivity sweep

```
  kept   out sz   iter    prealloc  branchless  bl_nozero  neon   v2    v2_nozero  4lane  stnp
  -----  -------  ------  --------  ---------  ---------  ------  ----  ---------  ------  ----
    1%     9963    0.44     0.42     0.34     0.25     0.32     0.32     0.26     0.32     0.37
   25%   249097    1.70     1.60     0.29     0.23     0.31     0.32     0.26     0.32     0.40
   50%   499677    3.18     3.00     0.29     0.23     0.31     0.30     0.26     0.30     0.37
   75%   749217    1.81     1.59     0.30     0.24     0.32     0.32     0.27     0.30     0.36
   99%   989889    1.05     0.51     0.31     0.25     0.32     0.32     0.28     0.32     0.38
```

### Table 2 — shuffled vs sorted（同 `filter_iter`，50%）

```
  shuffled        3.08 ms
  sorted          0.61 ms
  speedup on sorted: ~5-8x
```

### Table 3 — memcpy 上限

```
  copy            0.17 ms   ← 任何 filter 都至少要读 + 写
```

### Summary（50% kept，稳定态）

```
  variant          time      vs iter   vs copy_ceil
  filter_iter        3.10 ms    1.00x    17.85x
  prealloc           3.00 ms    1.06x    17.25x
  branchless         0.29 ms   10.33x     1.73x
  branchless_nozero  0.23 ms   13.55x     1.31x   ← 复用 buffer 省 0.06 ms
  neon               0.31 ms   10.36x     1.77x
  neon_v2            0.30 ms   10.47x     1.75x
  neon_v2_nozero     0.26 ms   11.08x     1.61x   ← 复用 buffer 省 0.04 ms
  neon_4lane         0.30 ms   10.57x     1.73x
  stnp (NT pair)     0.38 ms    8.50x     2.15x   ← 反而更慢
```

## 关键发现

### 1. branchless 仍然是最重要的 10x 提速

`filter_iter` (3.10 ms) → `filter_branchless` (0.29 ms) = **10.7x**。50% 随机选择率下分支预测器等于掷硬币，每次错判 15-20 cycles × 50 万次 ≈ 2 ms 纯惩罚。

AArch64 上 `filter_branchless` 编译成 7 条标量指令：

```asm
ldr   d1, [x0], #8                    ; load
str   d1, [x20, x9, lsl #3]           ; store at out[n*8]
fcmp  d1, d0                          ; compare
cinc  x9, x9, gt                      ; n += (x > threshold)  ← AArch64 硬件 branchless!
subs  x19, x19, #8
b.ne  LBB3_5
```

- `cinc`（conditional increment）是 AArch64 的硬件级 branchless，**不是分支**
- `[x20, x9, lsl #3]` 是 scaled-index store，写 `out[n]` 不需要单独算偏移
- 7 条指令 / 元素，每个 ~1 cycle，**无预测失败**

### 2. 手工 SIMD 没有比标量 branchless 更快

`branchless` / `neon` / `neon_v2` / `neon_4lane` 在稳定态下都在 **0.29-0.32 ms**，差距 < 10%。LLVM 把标量 branchless 已经优化到 7 条指令的极致，**手工写 NEON 反而平均 8 条/元素**，被 SIMD lane 操作的开销抵消。

### 3. `stnp` 反而慢了 25%（这条路径的负向参照）

`stnp` 在所有 selectivity 下都比 `vst1q_f64` 慢 25%（0.38 vs 0.30 ms）。汇编里看出原因：

```asm
; vst1q_f64 路径（neon_v2，14 条指令/2 元素 = 7/元素）:
... compute mask, swap, bif, lsl offset ...
str   q2, [x9, x14]            ← 一条 q-register store

; stnp 路径（16 条指令/2 元素 = 8/元素）:
... compute mask, swap, bif, add offset ...
mov.d x15, v2[1]               ← 拆 q → 2x d 多出来的指令
fmov  x16, d2                  ← 同上
stnp  x16, x15, [x14]          ← the stnp itself
```

3 个原因叠加：

1. **q → 2x d 的 bit-cast 多 2 条指令**（asm! 不接受 SIMD 寄存器做 operand）。
2. **NT hint 在 Apple M4 上可能不被严格实现**（Apple Silicon 是自研微架构）。
3. **8 MB 输出缓冲正好压在 system cache 边界**（~16 MB），**没有 cache pressure 需要缓解**——用错场景。

**结论**：streaming store 在 Apple Silicon 上对 cache-resident 的中小缓冲**反而吃亏**。

### 4. 跳过零填充是个真实有效的优化（**新发现**）

这是这一轮真正的赢家：

| | time | vs zero-init |
|---|---:|---:|
| branchless | 0.29 ms | baseline |
| **branchless_nozero** | **0.23 ms** | **省 0.06 ms（-21%）** |
| neon_v2 | 0.30 ms | baseline |
| **neon_v2_nozero** | **0.26 ms** | **省 0.04 ms（-13%）** |

**原理**：`vec![0.0; N]` 的 8 MB memset 占用了 ~0.05-0.07 ms 的内存带宽（按 ~150 GB/s 计算）。branchless 循环保证 `out[n]` 总被写入，所以 `[0, n_final)` 全是真实值，`truncate(n_final)` 直接丢弃未初始化的尾部——完全 sound。

**新极限**：`branchless_nozero` 跑到 **0.23 ms = 1.31x copy_ceiling**（之前是 1.66x）。我们已经穿过 "memory-bound 的最后一公里" 大半，剩下 ~0.05 ms 是 branchless 循环本身的 compare + select + conditional advance 开销。

### 5. 全栈优化路径

把 4 个优化叠加起来的累计效果（vs `filter_iter` 3.10 ms）：

| 优化 | 50% time | 累计提速 | vs copy_ceil |
|---|---:|---:|---:|
| `filter_iter` (baseline) | 3.10 ms | 1.0x | 17.85x |
| `branchless` | 0.29 ms | **10.7x** | 1.73x |
| `branchless_nozero` | 0.23 ms | **13.5x** | 1.31x |
| （理论极限 `memcpy`） | 0.17 ms | 18.2x | 1.00x |

## 与文章结论对照

| 文章现象 | 复现里 |
|---|---|
| iter 在 50% 最慢 | ✅ 3.10 ms |
| branchless 列几乎水平 | ✅ 0.29-0.32 ms |
| sorted 比 shuffled 快 ~4.5x | ✅ **~5-8x** |
| iter 在 1% 胜过 branchless | ⚠️ M4 上 branchless 反胜 |
| "almost 4x"（iter 50% vs branchless 50%） | ✅ **~10x** |
| （文章没有的）streaming store | ❌ M4 + 8 MB 上**负优化** |
| （文章没有的）跳过零填充 | ✅ **额外 -21%**，到 1.31x copy_ceil |

## Takeaway

1. **分支本身不贵，预测失败的分支才贵**。10x 提速全靠 branchless。
2. **Branchless 把控制依赖变数据依赖**（AArch64 上是 `cinc`）。
3. **LLVM 是非常强的标量优化器**：`cinc` + scaled-index 已经把标量 branchless 压到 7 条指令/元素的极限，手工 SIMD 没空间。
4. **到 memory-bound 阶段，再优化边际收益从零开始变为负**——`stnp` 反而慢 25%。
5. **跳过 `vec![0.0; N]` 的零填充是个真正有效的招**：8 MB memset 在 memory-bound 代码里是隐性大头。用了 `Vec::with_capacity` + unsafe `set_len` 后，又省了 21%。
6. **streaming store 的判定标准**：输出 > 系统缓存的 50% 才考虑 `stnp` / `movntpd`。Apple M4 的 8 MB 输出落在 cache-resident 区，NT hint 是负优化。
7. **真到 memory-bound 的最后一公里**：把 filter 跟下游融合避免写出中间结果，或者 in-place 更新输入 buffer（牺牲原数据）。

## 适用架构

- 代码在所有 Rust 支持的目标上都能编译；
- NEON 路径只在 `target_arch = "aarch64"` 时启用，其它架构自动 fallback 到 `filter_branchless`；
- `stnp` 通过 `core::arch::asm!` 直接发指令，无需 nightly / std feature；
- `Vec::set_len` 需要 `unsafe` block，但逻辑上 sound（f64 没有 Drop，branchless 循环保证写入）；
- 没有动态 CPU feature detection——aarch64 的 NEON 是基线。

## 复现性

- 数据集由 `SplitMix64` 用固定种子 `0xDEADBEEF_CAFEBABE` 生成。
- `bench_min` 跑 30 次取 min，丢弃前 3 次 warmup。
- 没有绑核 / `taskset` / 关 turbo 等控制；想更精确可加 `nice -n -20` + `sudo pmset -a disablesleep 1`。
- 第一次跑和后续跑有时差 ~2x（冷缓存 / OS 抖动）。建议至少跑 3 次取后两次的稳定值。

## C 版本

`c/` 下是同一套 benchmark 的 C 移植（`make run`），9 个实现一一对应。所有结论都复现了：
branchless ~11x、nozero 再省 ~20%、手工 SIMD 无显著优势、`stnp` 依然负优化。C 和 Rust
的数字落在同一区间（两边都是 LLVM 后端），说明这些结论是**微架构层面的**，与语言无关。

移植时踩的两个坑记在 `c/README.md` 里，都会悄悄给出好看但错误的数字：**dead store
elimination 把没人读的写循环整个删掉**（要用 `black_box` 挡住），以及**收缩 `realloc`
不等于 `Vec::truncate`**（前者在 macOS 上会 memcpy 搬走保留前缀，把省下的 memset 吃回去）。
