//! Reproducing "Branchless Rust: Making a Filter 4x Faster by Removing an if"
//! by Serhii Potapov (greyblake), 2026-08-02.
//!
//! https://www.greyblake.com/blog/branchless-rust/
//!
//! Compared side-by-side with an aarch64 NEON port of the branchless filter
//! (`filter_simd_neon`). Uses only `std`, no criterion — same shape of
//! benchmark (min over many iterations on 1M f64 values across several
//! selectivity levels).

use std::hint::black_box;
use std::time::Instant;

/// Size of the test buffer — matches the article.
const N: usize = 1_000_000;

// -----------------------------------------------------------------------------
// Deterministic RNG: splitmix64. Avoids pulling in `rand` and gives a fixed
// seed so results are reproducible across machines.
// -----------------------------------------------------------------------------
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    /// Uniformly distributed f64 in [0.0, 100.0).
    fn next_f64(&mut self) -> f64 {
        // Use the top 53 bits for the mantissa, just like the standard trick.
        let bits = self.next_u64() >> 11;
        let frac = bits as f64 / (1u64 << 53) as f64;
        frac * 100.0
    }
}

// -----------------------------------------------------------------------------
// Scalar filter implementations (the three from the article).
// -----------------------------------------------------------------------------

/// Idiomatic baseline: iterator + filter + collect.
pub fn filter_iter(input: &[f64], threshold: f64) -> Vec<f64> {
    input.iter().copied().filter(|&x| x > threshold).collect()
}

/// Pre-allocated capacity, but still branches on the threshold.
pub fn filter_prealloc(input: &[f64], threshold: f64) -> Vec<f64> {
    let mut out = Vec::with_capacity(input.len());
    for &x in input {
        if x > threshold {
            out.push(x);
        }
    }
    out
}

/// Branchless: always write, advance cursor only when the element is kept.
pub fn filter_branchless(input: &[f64], threshold: f64) -> Vec<f64> {
    let mut out = vec![0.0; input.len()];
    let mut n = 0usize;
    for &x in input {
        out[n] = x;
        n += (x > threshold) as usize;
    }
    out.truncate(n);
    out
}

/// Branchless + skip the zero-fill on the output buffer.
///
/// `vec![0.0; N]` memsets 8 MB to zero before we ever use it. The branchless
/// loop *always* writes `out[n] = x` for every input element, so by the time
/// we call `truncate(n_final)` every slot in `[0, n_final)` holds a real kept
/// value. The slots in `(n_final, N)` were never read — we can hand them to
/// the allocator as "initialized" via `set_len` and let `truncate` quietly
/// abandon them when we shrink. For `f64` (no `Drop`) this is sound.
///
/// Expected savings: the cost of zeroing 8 MB, ~0.05 ms on M4 — about 15-20%
/// of the current `filter_branchless` runtime.
pub fn filter_branchless_nozero(input: &[f64], threshold: f64) -> Vec<f64> {
    let mut out: Vec<f64> = Vec::with_capacity(input.len());
    // SAFETY: every slot in `[0, n_final)` will be written by the loop below
    // before we `truncate` to `n_final`. The remaining slots are abandoned
    // by `truncate` without ever being read; `f64` has no `Drop` impl, so
    // the allocator will reclaim the buffer unchanged.
    unsafe { out.set_len(input.len()) };

    let mut n = 0usize;
    for &x in input {
        // `n` is bounded by the number of elements processed so far, which
        // is always ≤ `input.len()`, so the index is in range.
        unsafe {
            *out.as_mut_ptr().add(n) = x;
        }
        n += (x > threshold) as usize;
    }
    out.truncate(n);
    out
}

// -----------------------------------------------------------------------------
// NEON port of the branchless filter.
//
// Reads two f64 at a time, asks NEON to produce the keep mask (`vcgtq_f64`),
// converts the mask bits to 0/1 lane values, then writes each lane
// individually at the current cursor — exactly the branchless trick, just
// with the compare done in SIMD.
// -----------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    /// NEON-enabled branchless filter. `unsafe` only because of the
    /// `target_feature` attribute; logic itself is correct as long as
    /// `input` is a valid slice.
    #[target_feature(enable = "neon")]
    #[allow(unsafe_op_in_unsafe_fn)] // raw-pointer ops on the input/output
                                     // buffer are sound: bounds checked by
                                     // construction below.
    pub unsafe fn filter_simd_neon(input: &[f64], threshold: f64) -> Vec<f64> {
        let mut out = vec![0.0_f64; input.len()];
        let mut n = 0usize;
        let vth = vdupq_n_f64(threshold);
        let chunks = input.len() / 2;

        let in_ptr = input.as_ptr();
        let out_ptr = out.as_mut_ptr();

        for i in 0..chunks {
            unsafe {
                // Load 2 f64 in one instruction.
                let lo = vld1q_f64(in_ptr.add(2 * i));
                // Per-lane keep mask: all-ones if keep, all-zeros if skip.
                // `vcgtq_f64` already returns `uint64x2_t` on aarch64.
                let keep_mask = vcgtq_f64(lo, vth);
                // Shift right by 63 to collapse each lane to 0 or 1.
                let one_bits = vshrq_n_u64(keep_mask, 63);

                let m0 = vgetq_lane_u64(one_bits, 0) as usize;
                let m1 = vgetq_lane_u64(one_bits, 1) as usize;

                let a = vgetq_lane_f64(lo, 0);
                let b = vgetq_lane_f64(lo, 1);

                // Branchless writes at the current cursor — same trick as
                // scalar, just driven by SIMD-produced increments.
                *out_ptr.add(n) = a;
                n += m0;
                *out_ptr.add(n) = b;
                n += m1;
            }
        }

        // Tail (0 or 1 element since N is even).
        for j in (chunks * 2)..input.len() {
            let x = *in_ptr.add(j);
            *out_ptr.add(n) = x;
            n += (x > threshold) as usize;
        }

        out.truncate(n);
        out
    }

    // -------------------------------------------------------------------------
    // V2: in-register compaction + single vector store per 2-lane block.
    //
    // Idea: instead of writing the two source lanes [a, b] separately (with
    // a `vgetq_lane_*` for each), produce a compacted vector
    //     compact = [first_kept, second_kept or garbage]
    // using a lane swap (`vextq_f64` to get [b, a]) plus a bitwise select
    // (`vbslq_f64` chooses per-lane between `lo` and `swap`), and emit the
    // whole pair with one `vst1q_f64`. The popcount `vaddvq_u64(one_bits)`
    // tells us how many real elements we just wrote, so `n` advances by that.
    //
    // The "second slot" can be garbage (e.g. when only `a` was kept we end
    // up writing `a` twice into [n, n+1]); that's harmless because either
    // the next iteration overwrites it or the final `truncate(n)` cuts it
    // off.
    // -------------------------------------------------------------------------
    #[target_feature(enable = "neon")]
    #[allow(unsafe_op_in_unsafe_fn)]
    pub unsafe fn filter_simd_neon_v2(input: &[f64], threshold: f64) -> Vec<f64> {
        let mut out = vec![0.0_f64; input.len()];
        let mut n = 0usize;
        let vth = vdupq_n_f64(threshold);
        let chunks = input.len() / 2;

        let in_ptr = input.as_ptr();
        let out_ptr = out.as_mut_ptr();

        for i in 0..chunks {
            unsafe {
                let lo = vld1q_f64(in_ptr.add(2 * i));
                let keep_mask = vcgtq_f64(lo, vth);              // [0xFF.. or 0]
                let one_bits = vshrq_n_u64(keep_mask, 63);       // [0 or 1, 0 or 1]

                // Swap lanes: [a, b] -> [b, a]
                let swapped = vextq_f64(lo, lo, 1);
                // Per-lane: if mask set, take from `lo`; else from `swapped`.
                //   lane 0: m_a ? a : b          (the "first kept" element)
                //   lane 1: m_b ? b : a          (valid only when m_b is set)
                let compact = vbslq_f64(keep_mask, lo, swapped);

                // One vector store writes 16 bytes regardless of popcount.
                vst1q_f64(out_ptr.add(n), compact);

                // popcount = number of real elements we just stored.
                let pc = vaddvq_u64(one_bits) as usize;
                n += pc;
            }
        }

        // Tail (0 or 1 element since N is even).
        for j in (chunks * 2)..input.len() {
            let x = *in_ptr.add(j);
            *out_ptr.add(n) = x;
            n += (x > threshold) as usize;
        }

        out.truncate(n);
        out
    }

    // -------------------------------------------------------------------------
    // V2-nozero: V2 but skip the `vec![0.0; N]` zero-fill. Saves ~0.05 ms
    // of memset on M4 for 8 MB output.
    // -------------------------------------------------------------------------
    #[target_feature(enable = "neon")]
    #[allow(unsafe_op_in_unsafe_fn)]
    pub unsafe fn filter_simd_neon_v2_nozero(input: &[f64], threshold: f64) -> Vec<f64> {
        let mut out: Vec<f64> = Vec::with_capacity(input.len());
        // SAFETY: see `filter_branchless_nozero` — every slot in `[0, n)`
        // gets written by the loop; `truncate(n)` discards the rest without
        // reading, and `f64` has no `Drop`.
        unsafe { out.set_len(input.len()) };

        let mut n = 0usize;
        let vth = vdupq_n_f64(threshold);
        let chunks = input.len() / 2;

        let in_ptr = input.as_ptr();
        let out_ptr = out.as_mut_ptr();

        for i in 0..chunks {
            unsafe {
                let lo = vld1q_f64(in_ptr.add(2 * i));
                let keep_mask = vcgtq_f64(lo, vth);
                let one_bits = vshrq_n_u64(keep_mask, 63);

                let swapped = vextq_f64(lo, lo, 1);
                let compact = vbslq_f64(keep_mask, lo, swapped);

                vst1q_f64(out_ptr.add(n), compact);

                let pc = vaddvq_u64(one_bits) as usize;
                n += pc;
            }
        }

        // Tail.
        for j in (chunks * 2)..input.len() {
            let x = *in_ptr.add(j);
            *out_ptr.add(n) = x;
            n += (x > threshold) as usize;
        }

        out.truncate(n);
        out
    }

    // -------------------------------------------------------------------------
    // V2-stnp: same in-register compaction as V2, but uses `stnp` (store
    // non-temporal pair) instead of `vst1q_f64`. The CPU is hinted that the
    // 16-byte block we're writing is "write-once, won't be read soon" — so
    // it can bypass the cache hierarchy and free cache bandwidth for the
    // sequential input read.
    //
    // `stnp` is not exposed as a typed NEON intrinsic in stable Rust, so we
    // emit it via `core::arch::asm!`. We bit-cast the d-register halves to
    // u64 — `stnp` only cares about the 64-bit pattern, not the type.
    // -------------------------------------------------------------------------
    #[target_feature(enable = "neon")]
    #[allow(unsafe_op_in_unsafe_fn)]
    pub unsafe fn filter_simd_neon_stnp(input: &[f64], threshold: f64) -> Vec<f64> {
        let mut out = vec![0.0_f64; input.len()];
        let mut n = 0usize;
        let vth = vdupq_n_f64(threshold);
        let chunks = input.len() / 2;

        let in_ptr = input.as_ptr();
        let out_ptr = out.as_mut_ptr();

        for i in 0..chunks {
            unsafe {
                let lo = vld1q_f64(in_ptr.add(2 * i));
                let mask = vcgtq_f64(lo, vth);
                let one_bits = vshrq_n_u64(mask, 63);

                let swapped = vextq_f64(lo, lo, 1);
                let compact = vbslq_f64(mask, lo, swapped);

                // Bit-cast d-register halves to u64 for the asm call.
                let lo_u64: u64 = vget_lane_u64(vreinterpret_u64_f64(vget_low_f64(compact)), 0);
                let hi_u64: u64 = vget_lane_u64(vreinterpret_u64_f64(vget_high_f64(compact)), 0);

                // Non-temporal pair store: hint the CPU that these 16 bytes
                // are write-once, no later reuse, please skip the cache.
                core::arch::asm!(
                    "stnp {lo}, {hi}, [{base}]",
                    lo = in(reg) lo_u64,
                    hi = in(reg) hi_u64,
                    base = in(reg) out_ptr.add(n),
                    options(nostack, preserves_flags)
                );

                let pc = vaddvq_u64(one_bits) as usize;
                n += pc;
            }
        }

        // Tail.
        for j in (chunks * 2)..input.len() {
            let x = *in_ptr.add(j);
            *out_ptr.add(n) = x;
            n += (x > threshold) as usize;
        }

        out.truncate(n);
        out
    }

    // -------------------------------------------------------------------------
    // V3: same as V2 but processes 4 elements per iteration (two 2-lane blocks
    // interleaved). Halves the loop overhead, which matters when each
    // iteration is so small that the loop counter / branch on `i < chunks`
    // becomes a noticeable fraction of the cost.
    // -------------------------------------------------------------------------
    #[target_feature(enable = "neon")]
    #[allow(unsafe_op_in_unsafe_fn)]
    pub unsafe fn filter_simd_neon_4lane(input: &[f64], threshold: f64) -> Vec<f64> {
        let mut out = vec![0.0_f64; input.len()];
        let mut n = 0usize;
        let vth = vdupq_n_f64(threshold);
        let super_chunks = input.len() / 4;

        let in_ptr = input.as_ptr();
        let out_ptr = out.as_mut_ptr();

        let mut i = 0;
        while i < super_chunks {
            unsafe {
                // Two independent 2-lane blocks.
                let lo1 = vld1q_f64(in_ptr.add(4 * i));
                let lo2 = vld1q_f64(in_ptr.add(4 * i + 2));

                let m1 = vcgtq_f64(lo1, vth);
                let m2 = vcgtq_f64(lo2, vth);
                let ob1 = vshrq_n_u64(m1, 63);
                let ob2 = vshrq_n_u64(m2, 63);

                let swap1 = vextq_f64(lo1, lo1, 1);
                let swap2 = vextq_f64(lo2, lo2, 1);
                let c1 = vbslq_f64(m1, lo1, swap1);
                let c2 = vbslq_f64(m2, lo2, swap2);

                let pc1 = vaddvq_u64(ob1) as usize;
                let pc2 = vaddvq_u64(ob2) as usize;

                // The two vector stores may target the same cache line; that's
                // fine — they will coalesce in the store buffer.
                vst1q_f64(out_ptr.add(n), c1);
                vst1q_f64(out_ptr.add(n + pc1), c2);

                n += pc1 + pc2;
                i += 1;
            }
        }

        // Tail: 0..3 leftover elements after the last 4-wide chunk.
        let processed = super_chunks * 4;
        for j in processed..input.len() {
            let x = *in_ptr.add(j);
            *out_ptr.add(n) = x;
            n += (x > threshold) as usize;
        }

        out.truncate(n);
        out
    }
}

/// Public, safe entry point. On aarch64 it calls the NEON path, otherwise it
/// falls back to the scalar branchless implementation so the program still
/// builds (and runs correctly) on other architectures.
pub fn filter_simd_neon(input: &[f64], threshold: f64) -> Vec<f64> {
    #[cfg(target_arch = "aarch64")]
    {
        // NEON is mandatory on aarch64, so this unsafe is sound in practice.
        unsafe { neon::filter_simd_neon(input, threshold) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        filter_branchless(input, threshold)
    }
}

pub fn filter_simd_neon_v2(input: &[f64], threshold: f64) -> Vec<f64> {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon::filter_simd_neon_v2(input, threshold) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        filter_branchless(input, threshold)
    }
}

pub fn filter_simd_neon_v2_nozero(input: &[f64], threshold: f64) -> Vec<f64> {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon::filter_simd_neon_v2_nozero(input, threshold) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        filter_branchless_nozero(input, threshold)
    }
}

pub fn filter_simd_neon_stnp(input: &[f64], threshold: f64) -> Vec<f64> {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon::filter_simd_neon_stnp(input, threshold) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        filter_branchless(input, threshold)
    }
}

pub fn filter_simd_neon_4lane(input: &[f64], threshold: f64) -> Vec<f64> {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon::filter_simd_neon_4lane(input, threshold) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        filter_branchless(input, threshold)
    }
}

// -----------------------------------------------------------------------------
// A memcpy "ceiling" — how fast can we just copy N f64s? Useful as a
// reference: a perfect in-register SIMD filter can't go faster than this.
// -----------------------------------------------------------------------------
pub fn copy_ceiling(input: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0_f64; input.len()];
    out.copy_from_slice(input);
    out
}

// -----------------------------------------------------------------------------
// Benchmarking helpers.
// -----------------------------------------------------------------------------

type FilterFn = fn(&[f64], f64) -> Vec<f64>;

/// Run `iters` repetitions of `f(input, threshold)`, discard the first
/// `warmup`, and return the minimum elapsed time in milliseconds.
fn bench_min(f: FilterFn, input: &[f64], threshold: f64, warmup: usize, iters: usize) -> f64 {
    for _ in 0..warmup {
        let v = f(black_box(input), threshold);
        black_box(v);
    }
    let mut best = f64::INFINITY;
    for _ in 0..iters {
        let t0 = Instant::now();
        let v = f(black_box(input), threshold);
        black_box(&v as *const _);
        let dt = t0.elapsed().as_secs_f64() * 1e3;
        if dt < best {
            best = dt;
        }
    }
    best
}

/// Same shape but for the no-threshold copy ceiling.
fn bench_min_copy(input: &[f64], warmup: usize, iters: usize) -> f64 {
    for _ in 0..warmup {
        let v = copy_ceiling(black_box(input));
        black_box(v);
    }
    let mut best = f64::INFINITY;
    for _ in 0..iters {
        let t0 = Instant::now();
        let v = copy_ceiling(black_box(input));
        black_box(&v as *const _);
        let dt = t0.elapsed().as_secs_f64() * 1e3;
        if dt < best {
            best = dt;
        }
    }
    best
}

// -----------------------------------------------------------------------------
// Output helpers.
// -----------------------------------------------------------------------------

fn print_row(label: &str, time_ms: f64) {
    println!("  {:<10}  {:>8.2} ms", label, time_ms);
}

fn print_table_row(
    kept_pct: u32,
    out_size: usize,
    iter_ms: f64,
    prealloc_ms: f64,
    branchless_ms: f64,
    bl_nozero_ms: f64,
    simd_ms: f64,
    simd_v2_ms: f64,
    simd_v2_nozero_ms: f64,
    simd_4lane_ms: f64,
    stnp_ms: f64,
) {
    println!(
        "  {:>3}%   {:>6}  {:>6.2}   {:>6.2}   {:>6.2}   {:>6.2}   {:>6.2}   {:>6.2}   {:>6.2}   {:>6.2}   {:>6.2}",
        kept_pct, out_size, iter_ms, prealloc_ms, branchless_ms, bl_nozero_ms, simd_ms, simd_v2_ms, simd_v2_nozero_ms, simd_4lane_ms, stnp_ms
    );
}

// -----------------------------------------------------------------------------
// main
// -----------------------------------------------------------------------------

fn main() {
    println!("=== Branchless Rust — reproduction on this machine ===");
    println!("Target: {} ({})", std::env::consts::ARCH, std::env::consts::OS);
    println!("N = {} f64 values, uniformly distributed in [0.0, 100.0)", N);
    println!();

    // ---- Generate a deterministic input once ----
    let mut rng = SplitMix64::new(0xDEADBEEF_CAFEBABE);
    let input: Vec<f64> = (0..N).map(|_| rng.next_f64()).collect();

    // ---- Verify correctness ----
    // The NEON path must produce exactly the same output as the scalar
    // branchless path for every threshold.
    let cases: &[(u32, &str, f64)] = &[
        (1, "1%", 99.0),
        (25, "25%", 75.0),
        (50, "50%", 50.0),
        (75, "75%", 25.0),
        (99, "99%", 1.0),
    ];
    for &(_, label, threshold) in cases {
        let ref_out = filter_branchless(&input, threshold);
        let bl_nozero_out = filter_branchless_nozero(&input, threshold);
        let simd_out = filter_simd_neon(&input, threshold);
        let simd_v2_out = filter_simd_neon_v2(&input, threshold);
        let simd_v2_nozero_out = filter_simd_neon_v2_nozero(&input, threshold);
        let simd_4l_out = filter_simd_neon_4lane(&input, threshold);
        let stnp_out = filter_simd_neon_stnp(&input, threshold);
        for (variant, out) in [
            ("bl_nozero", &bl_nozero_out),
            ("neon", &simd_out),
            ("neon_v2", &simd_v2_out),
            ("neon_v2_nozero", &simd_v2_nozero_out),
            ("neon_4lane", &simd_4l_out),
            ("stnp", &stnp_out),
        ] {
            assert_eq!(
                ref_out.len(),
                out.len(),
                "[{} / {}] length mismatch: scalar={}, simd={}",
                label,
                variant,
                ref_out.len(),
                out.len()
            );
            assert_eq!(ref_out[0], out[0], "[{} / {}] first element", label, variant);
            assert_eq!(
                ref_out[ref_out.len() / 2],
                out[out.len() / 2],
                "[{} / {}] middle element",
                label,
                variant
            );
            assert_eq!(
                ref_out[ref_out.len() - 1],
                out[out.len() - 1],
                "[{} / {}] last element",
                label,
                variant
            );
        }
    }
    println!("Correctness: NEON variants match scalar branchless for all cases.");
    println!();

    // ---- Table 1: selectivity sweep, shuffled data ----
    println!("Table 1 \u{2014} selectivity sweep on shuffled random data");
    println!("  kept   out sz   iter    prealloc  branchless  bl_nozero  neon   v2    v2_nozero  4lane  stnp");
    println!("  -----  -------  ------  --------  ---------  ---------  ------  ----  ---------  ------  ----");

    let warmup = 3;
    let iters = 30;

    let mut results: Vec<(
        u32,
        usize,
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
    )> = Vec::new();

    for &(kept_pct, _label, threshold) in cases {
        let out = filter_branchless(&input, threshold);
        let out_size = out.len();

        let t_iter = bench_min(filter_iter, &input, threshold, warmup, iters);
        let t_pre = bench_min(filter_prealloc, &input, threshold, warmup, iters);
        let t_bl = bench_min(filter_branchless, &input, threshold, warmup, iters);
        let t_bl_nozero = bench_min(filter_branchless_nozero, &input, threshold, warmup, iters);
        let t_simd = bench_min(filter_simd_neon, &input, threshold, warmup, iters);
        let t_simd_v2 = bench_min(filter_simd_neon_v2, &input, threshold, warmup, iters);
        let t_simd_v2_nozero =
            bench_min(filter_simd_neon_v2_nozero, &input, threshold, warmup, iters);
        let t_simd_4l = bench_min(filter_simd_neon_4lane, &input, threshold, warmup, iters);
        let t_stnp = bench_min(filter_simd_neon_stnp, &input, threshold, warmup, iters);

        print_table_row(
            kept_pct,
            out_size,
            t_iter,
            t_pre,
            t_bl,
            t_bl_nozero,
            t_simd,
            t_simd_v2,
            t_simd_v2_nozero,
            t_simd_4l,
            t_stnp,
        );
        results.push((
            kept_pct,
            out_size,
            t_iter,
            t_pre,
            t_bl,
            t_bl_nozero,
            t_simd,
            t_simd_v2,
            t_simd_v2_nozero,
            t_simd_4l,
            t_stnp,
        ));
    }
    println!();

    // ---- Table 2: the "smoking gun" \u2014 shuffled vs sorted at 50% ----
    println!("Table 2 \u{2014} shuffled vs sorted, same code (filter_iter), 50% kept");
    let mut sorted = input.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let t_shuf = bench_min(filter_iter, &input, 50.0, warmup, iters);
    let t_sort = bench_min(filter_iter, &sorted, 50.0, warmup, iters);
    print_row("shuffled", t_shuf);
    print_row("sorted", t_sort);
    println!("  speedup on sorted: {:.2}x", t_shuf / t_sort);
    println!();

    // ---- Table 3: copy ceiling ----
    println!("Table 3 \u{2014} memcpy ceiling (how fast can we just copy N f64?)");
    let t_copy = bench_min_copy(&input, warmup, iters);
    print_row("copy", t_copy);
    println!();

    // ---- Summary ----
    println!("Summary (speedup vs `filter_iter` at 50% kept)");
    let fifty = results
        .iter()
        .find(|(k, _, _, _, _, _, _, _, _, _, _)| *k == 50)
        .unwrap();
    let (kept, _, t_iter, t_pre, t_bl, t_bl_nozero, t_simd, t_v2, t_v2_nozero, t_4l, t_stnp) =
        fifty;
    println!("  variant          time      vs iter   vs copy_ceil");
    println!(
        "  filter_iter        {:>7.2} ms   1.00x     {:.2}x",
        t_iter,
        t_iter / t_copy
    );
    println!(
        "  prealloc           {:>7.2} ms   {:.2}x     {:.2}x",
        t_pre,
        t_iter / t_pre,
        t_pre / t_copy
    );
    println!(
        "  branchless         {:>7.2} ms   {:.2}x     {:.2}x",
        t_bl,
        t_iter / t_bl,
        t_bl / t_copy
    );
    println!(
        "  branchless_nozero  {:>7.2} ms   {:.2}x     {:.2}x",
        t_bl_nozero,
        t_iter / t_bl_nozero,
        t_bl_nozero / t_copy
    );
    println!(
        "  neon               {:>7.2} ms   {:.2}x     {:.2}x",
        t_simd,
        t_iter / t_simd,
        t_simd / t_copy
    );
    println!(
        "  neon_v2            {:>7.2} ms   {:.2}x     {:.2}x",
        t_v2,
        t_iter / t_v2,
        t_v2 / t_copy
    );
    println!(
        "  neon_v2_nozero     {:>7.2} ms   {:.2}x     {:.2}x",
        t_v2_nozero,
        t_iter / t_v2_nozero,
        t_v2_nozero / t_copy
    );
    println!(
        "  neon_4lane         {:>7.2} ms   {:.2}x     {:.2}x",
        t_4l,
        t_iter / t_4l,
        t_4l / t_copy
    );
    println!(
        "  stnp (NT pair)     {:>7.2} ms   {:.2}x     {:.2}x",
        t_stnp,
        t_iter / t_stnp,
        t_stnp / t_copy
    );
    println!();
    let _ = kept;
}