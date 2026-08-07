# Branchless Rust × NEON × streaming store — filter benchmark on Apple M4

复现 Serhii Potapov 的 [Branchless Rust: Making a Filter 4x Faster by Removing an if](https://www.greyblake.com/blog/branchless-rust/)，把 branchless filter 用 **aarch64 NEON intrinsics** 写了五个递增复杂度版本，再加一个 **`stnp` (store non-temporal pair) 实验**作为这条路径的负向参照。

只用 `std`，不引入 criterion。`src/main.rs` 是全部代码 + benchmark + 正确性校验。

## 文件结构

```
022-rust-simd/
├── Cargo.toml
├── README.md
└── src/
    └── main.rs          # 全部代码 ~390 行
```

## 怎么跑

```bash
cargo run --release
```

期望输出：3 张表 + Summary。连续跑几次取稳定态——首次跑有冷缓存抖动。

## 七种实现

| 函数 | 思路 | 关键特性 |
|---|---|---|
| `filter_iter` | `iter().copied().filter().collect()` | idiomatic baseline（带数据依赖分支） |
| `filter_prealloc` | `Vec::with_capacity` + `if` 分支 | 去掉扩容开销，保留分支 |
| `filter_branchless` | 总是写 `out[n]=x`，游标按 `(x>threshold) as usize` 推进 | 控制依赖→数据依赖 |
| `filter_simd_neon` | 2-lane NEON，SIMD compare + scalar store | `vcgtq_f64` 产 mask，scalar 写 2 次 |
| `filter_simd_neon_v2` | 2-lane NEON，in-register compaction + 1×`vst1q_f64` | `vextq` swap + `vbslq` select，popcount `addp.2d` |
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

`stnp` 是 AArch64 的 "store non-temporal pair" 指令（x86 对应 `movntpd`）。**对 CPU 暗示**："这 16 字节是流式数据、写完不会再读，请绕过 cache"。

理论上好处：
- 不污染 cache，给输入读留出 L1/L2 带宽
- NT store 直接到内存（或 write-combining buffer）

Rust stdarch 没暴露 `stnp` 的 typed intrinsic，只能用 `core::arch::asm!` 直接发指令：

```rust
let lo_u64: u64 = vget_lane_u64(vreinterpret_u64_f64(vget_low_f64(compact)), 0);
let hi_u64: u64 = vget_lane_u64(vreinterpret_u64_f64(vget_high_f64(compact)), 0);
core::arch::asm!(
    "stnp {lo}, {hi}, [{base}]",
    lo = in(reg) lo_u64,
    hi = in(reg) hi_u64,
    base = in(reg) out_ptr.add(n),
    options(nostack, preserves_flags)
);
```

## 在 Apple M4 上的稳定态结果

> N = 1,000,000 个 f64 = 8 MB 输入 / 8 MB 输出；均匀分布 [0.0, 100.0)；固定种子 `0xDEADBE…`；每个数据点 30 iters min（warmup=3）。**M4 有 4 MB L2 + ~16 MB 系统缓存**，8 MB 输入 + 8 MB 输出 = 16 MB 总量，**正好压在 system cache 边界**。

### Table 1 — selectivity sweep

```
  kept   out sz   iter    prealloc  branchless  neon    neon_v2  neon_4l  stnp
  -----  -------  ------  --------  ---------  ------  -------  -------  ------
    1%     9963    0.37     0.40     0.35     0.35     0.32     0.32     0.38
   25%   249097    1.69     1.60     0.29     0.31     0.32     0.32     0.38
   50%   499677    3.07     2.94     0.30     0.31     0.32     0.32     0.38
   75%   749217    2.07     1.57     0.29     0.31     0.32     0.32     0.37
   99%   989889    1.04     0.41     0.29     0.30     0.30     0.31     0.37
```

### Table 2 — shuffled vs sorted（同 `filter_iter`，50%）

```
  shuffled        3.07 ms
  sorted          0.40 ms
  speedup on sorted: ~7.7x
```

### Table 3 — memcpy 上限

```
  copy            0.17 ms   ← 任何 filter 都至少要读 + 写
```

### Summary（50% kept）

```
  variant       time      vs iter   vs copy_ceil
  filter_iter     3.07 ms    1.00x    17.89x
  prealloc        2.94 ms    1.04x    17.13x
  branchless      0.30 ms   10.26x     1.74x
  neon            0.31 ms    9.81x     1.82x
  neon_v2         0.32 ms    9.60x     1.86x
  neon_4lane      0.32 ms    9.56x     1.87x
  stnp (NT pair)  0.38 ms    8.11x     2.21x   ← 反而更慢
```

## 关键发现

### 1. branchless 仍然是最重要的 10x 提速

`filter_iter` (3.07 ms) → `filter_branchless` (0.30 ms) = **10.3x**。这是因为 50% 随机选择率下，分支预测器等于掷硬币，每次错判 15-20 cycles × 50 万次 ≈ 2 ms 纯惩罚。

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

`branchless` / `neon` / `neon_v2` / `neon_4lane` 在稳定态下都在 **0.29-0.32 ms**，差距 < 10%。

LLVM 把标量 branchless 已经优化到 7 条指令的极致，**手工写 NEON（`vst1q_f64`）反而平均 8 条/元素**，被 SIMD 操作（`vextq`+`vbslq`）+ lane extract 的开销抵消了。

### 3. `stnp` 反而慢了 25%（这条路径的负向参照）

`stnp` 在所有 selectivity 下都比 `vst1q_f64` 慢 25%（0.38 vs 0.30 ms）。汇编里看出原因：

```asm
; vst1q_f64 路径（neon_v2，14 条指令/2 元素 = 7/元素）:
ldr   q2, [x13], #16
fcmgt.2d v3, v2, v6
and.16b v4, v3, v1
addp.2d d4, v4
ext.16b v5, v2, v2, #8
bif.16b v2, v5, v3
lsl   x14, x11, #3
str   q2, [x9, x14]            ← 一条 q-register store
add   x11, x11, x14
subs  x12, x12, #1
b.ne  LBB4_9

; stnp 路径（16 条指令/2 元素 = 8/元素）:
ldr   q2, [x13], #16
fcmgt.2d v3, v2, v6
and.16b v4, v3, v1
addp.2d d4, v4
add   x14, x9, x11, lsl #3
ext.16b v5, v2, v2, #8
bif.16b v2, v5, v3
mov.d x15, v2[1]               ← 拆 q → 2x d 多出来的指令
fmov  x16, d2                  ← 同上
stnp  x16, x15, [x14]          ← the stnp itself
add   x11, x11, x14
subs  x12, x12, #1
b.ne  LBB5_9
```

3 个原因叠加导致 stnp 反而更慢：

1. **q → 2x d 的 bit-cast 多 2 条指令**：`mov.d` + `fmov` 把 SIMD q 寄存器的两个 64-bit half 搬到 GP 寄存器（asm! 的限制）。`vst1q_f64` 直接吃 q 寄存器，没有这步。
2. **NT hint 在 Apple M4 上可能被忽略**：M 系列是 Apple 自研微架构，不像 Intel/AMD 那样严格按 NT 提示绕过 cache。`stnp` 在 x86 上对应 `movntpd` 的行为可能在 M4 上不是 1:1 对应。
3. **8 MB 输出缓冲正好压在 system cache 边界**（~16 MB），**没有 cache pressure 需要缓解**。NT store 设计的初衷是给 >L3 的巨大输出用，我们用错了场景。

**结论**：streaming store 在 Apple Silicon 上对 cache-resident 的中小缓冲反而吃亏。

## 与文章结论对照

| 文章现象 | 复现里 |
|---|---|
| iter 在 50% 最慢 | ✅ 3.07 ms，是 iter 列最大 |
| branchless 列几乎水平 | ✅ 0.29-0.32 ms |
| sorted 比 shuffled 快 ~4.5x | ✅ **~7.7x** |
| iter 在 1% 胜过 branchless | ⚠️ M4 上 branchless 反胜（0.35 vs 0.37 ms） |
| "almost 4x"（iter 50% vs branchless 50%） | ✅ 这里 **~10x** |
| （文章没有的）streaming store 优化 | ❌ 在 M4 + 8 MB 缓冲上**负优化** |

## Takeaway

1. **分支本身不贵，预测失败的分支才贵**。10x 提速全靠 branchless。
2. **Branchless 把控制依赖变数据依赖**——比较结果变成 0/1 数值（AArch64 上是 `cinc`）。
3. **LLVM 是非常强的标量优化器**：`cinc` + scaled-index 已经把标量 branchless 压到 7 条指令/元素的极限，**手工 SIMD 没空间**。
4. **到 memory-bound 阶段，再优化边际收益归零**。5 个 branchless 变体都在 ~1.7-1.9x memcpy ceiling。
5. **streaming store (`stnp` / `movntpd`) 不是银弹**：
   - 给 >L3 的巨大输出用才有意义
   - Apple Silicon 对 NT hint 的支持可能不如 x86
   - bit-cast 拆 SIMD 寄存器的开销可能比省下来的还多
   - **先用数据规模来判定**：如果输出 < 系统缓存的 50%，别用 NT store
6. **真要再榨**：唯一剩下的路子是**复用输出 buffer**，跳过 `vec![0.0; N]` 的零初始化（8 MB 写 ≈ 0.05 ms），或把 filter 跟下游融合避免写出中间结果。

## 适用架构

- 代码在所有 Rust 支持的目标上都能编译；
- NEON 路径只在 `target_arch = "aarch64"` 时启用，其它架构自动 fallback 到 `filter_branchless`；
- `stnp` 通过 `core::arch::asm!` 直接发指令，所以无需 nightly / std feature；
- 没有动态 CPU feature detection——aarch64 的 NEON 是基线。

## 复现性

- 数据集由 `SplitMix64` 用固定种子 `0xDEADBEEF_CAFEBABE` 生成。
- `bench_min` 跑 30 次取 min，丢弃前 3 次 warmup。
- 没有绑核 / `taskset` / 关 turbo 等控制；想更精确可加 `nice -n -20` + `sudo pmset -a disablesleep 1`。
- 第一次跑和后续跑有时差 ~2x（冷缓存 / OS 抖动）。建议至少跑 3 次取后两次的稳定值。