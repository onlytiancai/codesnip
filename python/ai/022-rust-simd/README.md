# Branchless Rust × NEON — filter benchmark on Apple M4

复现 Serhii Potapov 的文章 [Branchless Rust: Making a Filter 4x Faster by Removing an if](https://www.greyblake.com/blog/branchless-rust/)，并把同样的 branchless filter 用 **aarch64 NEON intrinsics** 实现了一版作为对照。

只用 `std`，不引入 criterion。`src/main.rs` 里是完整的 filter 实现 + benchmark + 正确性校验。

## 文件结构

```
022-rust-simd/
├── Cargo.toml
├── README.md
└── src/
    └── main.rs          # 全部代码 ~210 行
```

## 怎么跑

```bash
cargo run --release
```

期望输出：在 aarch64-apple-darwin 上跑出"Table 1 / 2 / 3"三张表 + Summary。

## 四种实现

| 函数 | 思路 | 关键特性 |
|---|---|---|
| `filter_iter` | `iter().copied().filter().collect()` | idiomatic baseline |
| `filter_prealloc` | 手动 `Vec::with_capacity` + `if` 分支 | 去掉扩容开销，但保留数据依赖分支 |
| `filter_branchless` | 总是写 `out[n]=x`，游标按 `(x>threshold) as usize` 推进 | 控制依赖→数据依赖，消除分支预测失败 |
| `filter_simd_neon` | 2-lane NEON 版 branchless：SIMD 比较 + scalar 写 | `vld1q_f64` / `vcgtq_f64` / `vshrq_n_u64` / `vgetq_lane_*` |

外加一个 **`copy_ceiling`**：`out.copy_from_slice(input)`，作为"光复制 N 个 f64"的内存带宽上限。

## NEON 实现要点

```rust
let lo = vld1q_f64(in_ptr.add(2 * i));     // 一次取 2 个 f64
let keep_mask = vcgtq_f64(lo, vth);        // 每 lane：全 1（保留）或全 0（丢弃）
let one_bits  = vshrq_n_u64(keep_mask, 63); // 每 lane 塌成 0 或 1
let m0 = vgetq_lane_u64(one_bits, 0) as usize;
let m1 = vgetq_lane_u64(one_bits, 1) as usize;
// 然后和标量 branchless 一样写 + 推进游标
```

本质是"用 SIMD 算比较、用 scalar 写"，NEON 的优势主要是 compare + mask collapse，store 还是 scalar。如果想要更激进，可以做 in-register compaction（`vbslq_f64` + lane permute）或者 streaming store，但那超出本文的复现范围。

## 在 Apple M4 上的实测结果

> N = 1,000,000 个 f64，均匀分布在 `[0.0, 100.0)`，固定种子 `0xDEADBE…`，每个数据点取 20 次 min（warmup=3）。

### Table 1 — selectivity sweep（shuffled 随机数据）

```
  kept     output size   iter        prealloc    branchless   neon
  -------- ------------  ---------   ---------   ---------   ---------
    1%          9963         0.60 ms        0.48 ms        0.38 ms        0.41 ms
   25%        249097         1.70 ms        1.61 ms        0.29 ms        0.31 ms
   50%        499677         3.12 ms        2.94 ms        0.29 ms        0.31 ms
   75%        749217         2.08 ms        1.59 ms        0.29 ms        0.31 ms
   99%        989889         1.00 ms        0.51 ms        0.29 ms        0.31 ms
```

### Table 2 — shuffled vs sorted（同一个 `filter_iter`，50%）

```
  shuffled        3.12 ms
  sorted          0.59 ms
  speedup on sorted: ~5.3x
```

### Table 3 — memcpy 上限

```
  copy            0.17 ms   ← 任何 filter 都至少要读 + 写
```

### 总结提速倍数（vs `filter_iter`）

```
  keep   1%  :  branchless  1.59x   neon  1.48x
  keep  25%  :  branchless  5.93x   neon  5.44x
  keep  50%  :  branchless 10.81x   neon 10.20x   ← 几乎打满内存带宽
  keep  75%  :  branchless  7.11x   neon  6.71x
  keep  99%  :  branchless  3.48x   neon  3.28x
```

## 与文章结论的对照

| 文章现象 | 复现里 |
|---|---|
| iter 在 50% 最慢 | ✅ 3.12 ms，是 iter 列最大 |
| branchless 列几乎水平 | ✅ 0.29–0.38 ms |
| sorted 比 shuffled 快 ~4.5x | ✅ **5.3x**（M4 更夸张） |
| iter 在 1% 胜过 branchless | ⚠️ M4 上 branchless 反胜（0.38 vs 0.60 ms），Intel 上的 trade-off 在这里被压垮了 |
| 文章给的"almost 4x"（iter 50% vs branchless 50%） | ✅ 这里 **10.8x** |

## 关键 takeaway

1. **分支本身不贵，预测失败的分支才贵**。50% 随机数据下预测器等于掷硬币，每次错判 15–20 cycles × 50 万次 ≈ 2 ms 的纯惩罚。
2. **Branchless 把"决定去哪儿"变成"算一个 0/1 当指针偏移"**，从控制依赖变成数据依赖，比较在汇编里就是一条 `seta` / `csel` 类指令。
3. **在 M4 上 memory subsystem 比 Intel i7-10875H 强很多**，所以绝对时间和相对倍数都被放大。文章里 branchless 50% 约 1 ms，这里 0.29 ms。
4. **当 branchless 已经接近 memcpy 上限时，再叠 SIMD 没有显著收益**——branchless 是 1.66x copy ceiling，NEON 是 1.77x copy ceiling。`vgetq_lane_*` + scalar 写相比纯标量 branchless 反而引入一点点开销。要再榨下去需要：
   - in-register compaction（`vbslq_f64` + lane permute），或
   - streaming / NT store（绕开 RFO），或
   - 更宽的 block（一次处理 4 个 f64），或
   - 输出 buffer 复用（避免 `vec![0.0; N]` 的零初始化）。
5. **除非 profiler 指着热路径，否则别上 branchless**——可读性差，最好情况变差。文章给的标准："branch on unpredictable data in a hot loop"。

## 适用架构

- 代码在所有 Rust 支持的目标上都能编译；
- NEON 路径只在 `target_arch = "aarch64"` 时启用，其它架构自动 fallback 到 `filter_branchless`。
- 没有用任何动态 CPU feature detection——aarch64 的 NEON 是基线，按规范保证存在。

## 复现性

- 数据集由 `SplitMix64` 用固定种子 `0xDEADBEEF_CAFEBABE` 生成。
- `bench_min` 每次跑固定 20 次取 min，丢弃前 3 次 warmup。
- 没有跨进程噪声控制（没绑核、`taskset` 等）；想更精确可以加 `nice -n -20` + 关闭 turbo boost。