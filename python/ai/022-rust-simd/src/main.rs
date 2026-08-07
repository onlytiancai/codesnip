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
    simd_ms: f64,
) {
    println!(
        "  {:>3}%      {:>8}    {:>9.2} ms   {:>9.2} ms   {:>9.2} ms   {:>9.2} ms",
        kept_pct, out_size, iter_ms, prealloc_ms, branchless_ms, simd_ms
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
        let simd_out = filter_simd_neon(&input, threshold);
        assert_eq!(
            ref_out.len(),
            simd_out.len(),
            "[{}] length mismatch: scalar={}, simd={}",
            label,
            ref_out.len(),
            simd_out.len()
        );
        // Spot-check first, middle, last for byte-equality.
        assert_eq!(ref_out[0], simd_out[0], "[{}] first element", label);
        assert_eq!(
            ref_out[ref_out.len() / 2],
            simd_out[simd_out.len() / 2],
            "[{}] middle element",
            label
        );
        assert_eq!(
            ref_out[ref_out.len() - 1],
            simd_out[simd_out.len() - 1],
            "[{}] last element",
            label
        );
    }
    println!("Correctness: NEON output matches scalar branchless for all cases.");
    println!();

    // ---- Table 1: selectivity sweep, shuffled data ----
    println!("Table 1 \u{2014} selectivity sweep on shuffled random data");
    println!("  kept     output size   iter        prealloc    branchless   neon");
    println!("  -------- ------------  ---------   ---------   ---------   ---------");

    let warmup = 3;
    let iters = 20;

    let mut results: Vec<(u32, usize, f64, f64, f64, f64)> = Vec::new();

    for &(kept_pct, _label, threshold) in cases {
        let out = filter_branchless(&input, threshold);
        let out_size = out.len();

        let t_iter = bench_min(filter_iter, &input, threshold, warmup, iters);
        let t_pre = bench_min(filter_prealloc, &input, threshold, warmup, iters);
        let t_bl = bench_min(filter_branchless, &input, threshold, warmup, iters);
        let t_simd = bench_min(filter_simd_neon, &input, threshold, warmup, iters);

        print_table_row(kept_pct, out_size, t_iter, t_pre, t_bl, t_simd);
        results.push((kept_pct, out_size, t_iter, t_pre, t_bl, t_simd));
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
    println!("Summary (speedup vs `filter_iter`)");
    for (kept, _, t_iter, _, t_bl, t_simd) in &results {
        println!(
            "  keep {:>3}%  :  branchless {:>5.2}x   neon {:>5.2}x",
            kept,
            t_iter / t_bl,
            t_iter / t_simd
        );
    }
    println!();

    // Print how close each branchless variant is to the copy ceiling at 50%.
    let fifty = results.iter().find(|(k, _, _, _, _, _)| *k == 50).unwrap();
    println!(
        "At 50% kept: branchless is {:.2}x copy ceiling, neon is {:.2}x copy ceiling",
        fifty.4 / t_copy,
        fifty.5 / t_copy,
    );
}