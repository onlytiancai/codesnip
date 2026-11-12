/*
 * branchless_filter.c — C port of the Rust branchless × NEON × stnp × nozero
 *                       filter benchmark on Apple M4.
 *
 * Reproduces:
 *   https://www.greyblake.com/blog/branchless-rust/
 *
 * Mirrors src/main.rs in the parent project, 9 implementations side by side:
 *   - filter_iter / filter_prealloc / filter_branchless (scalar baseline)
 *   - filter_branchless_nozero (skip the 8 MB zero-fill)
 *   - filter_simd_neon (2-lane NEON, scalar stores)
 *   - filter_simd_neon_v2 (in-register compaction + 1× vst1q)
 *   - filter_simd_neon_v2_nozero
 *   - filter_simd_neon_4lane (4-lane unrolled)
 *   - filter_simd_neon_stnp (stnp via inline asm)
 *   - copy_ceiling (memcpy ceiling)
 *
 * Build: see Makefile. Run: ./branchless_filter
 *
 * Only stdlib + arm_neon.h + inline asm. Same shape of correctness check
 * (3-point comparison vs scalar branchless) and same min-over-iters benchmark
 * as the Rust version.
 */

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__aarch64__)
#include <arm_neon.h>
#define HAVE_NEON 1
#else
#define HAVE_NEON 0
#endif

#if defined(__APPLE__)
#include <mach/mach_time.h>
#endif

/* Size of the test buffer — matches the article and the Rust port. */
#define N 1000000

/* ----------------------------------------------------------------------------
 * Deterministic RNG: splitmix64. Same constants as the Rust port so the
 * generated input is bit-identical to the Rust version's input.
 * -------------------------------------------------------------------------- */
typedef struct { uint64_t s; } splitmix64;

static inline uint64_t splitmix64_next(splitmix64 *r) {
    r->s += 0x9E3779B97F4A7C15ULL;
    uint64_t z = r->s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/* Uniformly distributed f64 in [0.0, 100.0). */
static inline double splitmix64_next_f64(splitmix64 *r) {
    /* Use the top 53 bits for the mantissa. */
    uint64_t bits = splitmix64_next(r) >> 11;
    double frac = (double)bits / (double)(1ULL << 53);
    return frac * 100.0;
}

/* ----------------------------------------------------------------------------
 * Scalar filter implementations — the three from the article.
 * -------------------------------------------------------------------------- */

/* Idiomatic baseline: walk + branch + push. */
static size_t filter_iter(const double *input, size_t n, double threshold,
                          double **out) {
    size_t cap = 4, len = 0;
    double *buf = (double *)malloc(cap * sizeof(double));
    if (!buf) { *out = NULL; return 0; }
    for (size_t i = 0; i < n; i++) {
        if (input[i] > threshold) {
            if (len == cap) {
                cap *= 2;
                double *nb = (double *)realloc(buf, cap * sizeof(double));
                if (!nb) { free(buf); *out = NULL; return 0; }
                buf = nb;
            }
            buf[len++] = input[i];
        }
    }
    *out = buf;
    return len;
}

/* Pre-allocated capacity, still branches on the threshold. */
static size_t filter_prealloc(const double *input, size_t n, double threshold,
                              double **out) {
    double *buf = (double *)malloc(n * sizeof(double));
    if (!buf) { *out = NULL; return 0; }
    size_t len = 0;
    for (size_t i = 0; i < n; i++) {
        if (input[i] > threshold) {
            buf[len++] = input[i];
        }
    }
    *out = buf;
    return len;
}

/* Branchless: always write, advance cursor only when the element is kept. */
static size_t filter_branchless(const double *input, size_t n, double threshold,
                                double **out) {
    double *buf = (double *)calloc(n, sizeof(double));
    if (!buf) { *out = NULL; return 0; }
    size_t cur = 0;
    for (size_t i = 0; i < n; i++) {
        double x = input[i];
        buf[cur] = x;
        cur += (x > threshold) ? 1 : 0;
    }
    *out = buf;
    return cur;
}

/*
 * Branchless + skip the zero-fill on the output buffer.
 *
 * `calloc` memsets 8 MB to zero before we ever use it; `malloc` hands back the
 * pages untouched. The branchless loop *always* writes `buf[cur] = x` for every
 * input element, so every slot in `[0, cur)` holds a real kept value by the time
 * we return. The slots in `[cur, n)` are still uninitialized, but we report the
 * length as `cur`, so nobody ever reads them. For `double` (no destructor) this
 * is sound — it mirrors Rust's `Vec::with_capacity` + `set_len` + `truncate`.
 *
 * Deliberately NOT realloc'ing down to `cur`: Rust's `truncate` only writes the
 * Vec's length field, whereas a shrinking `realloc` on macOS moves the block to
 * a smaller size class and memcpy's the retained prefix. That copy costs time
 * proportional to the kept count and swamps the memset we saved — it made this
 * variant *slower* than plain `calloc` and scale with selectivity. We keep the
 * full-size allocation and just return the length, exactly like the Rust port.
 */
static size_t filter_branchless_nozero(const double *input, size_t n,
                                       double threshold, double **out) {
    /* malloc + claim the capacity without zeroing; we write every slot we end
     * up keeping and simply report the shorter length. */
    double *buf = (double *)malloc(n * sizeof(double));
    if (!buf) { *out = NULL; return 0; }
    size_t cur = 0;
    for (size_t i = 0; i < n; i++) {
        double x = input[i];
        buf[cur] = x;
        cur += (x > threshold) ? 1 : 0;
    }
    *out = buf;
    return cur;
}

/* ----------------------------------------------------------------------------
 * NEON port of the branchless filter (aarch64 only).
 *
 * Reads two f64 at a time, asks NEON to produce the keep mask (vcgtq_f64),
 * converts the mask bits to 0/1 lane values, then writes each lane
 * individually at the current cursor — exactly the branchless trick, just
 * with the compare done in SIMD.
 * -------------------------------------------------------------------------- */

#if HAVE_NEON

/* NEON-enabled branchless filter. 2-lane, scalar stores. */
__attribute__((target("neon")))
static size_t filter_simd_neon(const double *input, size_t n, double threshold,
                               double **out) {
    double *buf = (double *)calloc(n, sizeof(double));
    if (!buf) { *out = NULL; return 0; }
    size_t cur = 0;
    float64x2_t vth = vdupq_n_f64(threshold);
    size_t chunks = n / 2;

    for (size_t i = 0; i < chunks; i++) {
        float64x2_t lo = vld1q_f64(input + 2 * i);
        uint64x2_t keep_mask = vcgtq_f64(lo, vth);             /* all-ones if keep */
        uint64x2_t one_bits = vshrq_n_u64(keep_mask, 63);      /* [0 or 1, 0 or 1] */

        size_t m0 = vgetq_lane_u64(one_bits, 0);
        size_t m1 = vgetq_lane_u64(one_bits, 1);

        double a = vgetq_lane_f64(lo, 0);
        double b = vgetq_lane_f64(lo, 1);

        /* Branchless writes at the current cursor — same trick as scalar. */
        buf[cur] = a;
        cur += m0;
        buf[cur] = b;
        cur += m1;
    }

    /* Tail (0 or 1 element since N is even). */
    for (size_t j = chunks * 2; j < n; j++) {
        double x = input[j];
        buf[cur] = x;
        cur += (x > threshold) ? 1 : 0;
    }

    *out = buf;
    return cur;
}

/*
 * V2: in-register compaction + single vector store per 2-lane block.
 * Produces a compacted vector [first_kept, second_kept_or_garbage] using a
 * lane swap (vextq_f64 to get [b, a]) plus a bitwise select (vbslq_f64 picks
 * per-lane between lo and swap), then emits the pair with one vst1q_f64.
 * The popcount vaddvq_u64 tells us how many real elements we just wrote.
 */
__attribute__((target("neon")))
static size_t filter_simd_neon_v2(const double *input, size_t n, double threshold,
                                  double **out) {
    double *buf = (double *)calloc(n, sizeof(double));
    if (!buf) { *out = NULL; return 0; }
    size_t cur = 0;
    float64x2_t vth = vdupq_n_f64(threshold);
    size_t chunks = n / 2;

    for (size_t i = 0; i < chunks; i++) {
        float64x2_t lo = vld1q_f64(input + 2 * i);
        uint64x2_t keep_mask = vcgtq_f64(lo, vth);             /* [0xFF.. or 0] */
        uint64x2_t one_bits = vshrq_n_u64(keep_mask, 63);      /* [0 or 1, 0 or 1] */

        /* Swap lanes: [a, b] -> [b, a] */
        float64x2_t swapped = vextq_f64(lo, lo, 1);
        /* Per-lane: if mask set, take from lo; else from swapped.
         *   lane 0: m_a ? a : b          (the "first kept" element)
         *   lane 1: m_b ? b : a          (valid only when m_b is set) */
        float64x2_t compact = vbslq_f64(keep_mask, lo, swapped);

        /* One vector store writes 16 bytes regardless of popcount. */
        vst1q_f64(buf + cur, compact);

        size_t pc = vaddvq_u64(one_bits);
        cur += pc;
    }

    /* Tail (0 or 1 element since N is even). */
    for (size_t j = chunks * 2; j < n; j++) {
        double x = input[j];
        buf[cur] = x;
        cur += (x > threshold) ? 1 : 0;
    }

    *out = buf;
    return cur;
}

/*
 * V2-nozero: V2 but skip the calloc zero-fill. Saves ~0.05 ms of memset on
 * M4 for the 8 MB output.
 */
__attribute__((target("neon")))
static size_t filter_simd_neon_v2_nozero(const double *input, size_t n,
                                         double threshold, double **out) {
    double *buf = (double *)malloc(n * sizeof(double));
    if (!buf) { *out = NULL; return 0; }
    size_t cur = 0;
    float64x2_t vth = vdupq_n_f64(threshold);
    size_t chunks = n / 2;

    for (size_t i = 0; i < chunks; i++) {
        float64x2_t lo = vld1q_f64(input + 2 * i);
        uint64x2_t keep_mask = vcgtq_f64(lo, vth);
        uint64x2_t one_bits = vshrq_n_u64(keep_mask, 63);

        float64x2_t swapped = vextq_f64(lo, lo, 1);
        float64x2_t compact = vbslq_f64(keep_mask, lo, swapped);

        vst1q_f64(buf + cur, compact);

        size_t pc = vaddvq_u64(one_bits);
        cur += pc;
    }

    /* Tail. */
    for (size_t j = chunks * 2; j < n; j++) {
        double x = input[j];
        buf[cur] = x;
        cur += (x > threshold) ? 1 : 0;
    }

    /* Report only the kept prefix; the tail stays uninitialized and unread.
     * No shrinking realloc here — see filter_branchless_nozero for why. */
    *out = buf;
    return cur;
}

/*
 * V2-stnp: same in-register compaction as V2, but uses stnp (store
 * non-temporal pair) instead of vst1q_f64. The CPU is hinted that the 16-byte
 * block we're writing is "write-once, won't be read soon" — so it can bypass
 * the cache hierarchy and free cache bandwidth for the sequential input read.
 *
 * stnp is not exposed as a typed NEON intrinsic in stable headers, so we emit
 * it via inline asm. We bit-cast the d-register halves to u64 — stnp only
 * cares about the 64-bit pattern, not the type.
 */
__attribute__((target("neon")))
static size_t filter_simd_neon_stnp(const double *input, size_t n, double threshold,
                                    double **out) {
    double *buf = (double *)calloc(n, sizeof(double));
    if (!buf) { *out = NULL; return 0; }
    size_t cur = 0;
    float64x2_t vth = vdupq_n_f64(threshold);
    size_t chunks = n / 2;

    for (size_t i = 0; i < chunks; i++) {
        float64x2_t lo = vld1q_f64(input + 2 * i);
        uint64x2_t mask = vcgtq_f64(lo, vth);
        uint64x2_t one_bits = vshrq_n_u64(mask, 63);

        float64x2_t swapped = vextq_f64(lo, lo, 1);
        float64x2_t compact = vbslq_f64(mask, lo, swapped);

        /* Bit-cast d-register halves to u64 for the asm call. */
        uint64_t lo_u64 = vget_lane_u64(vreinterpret_u64_f64(vget_low_f64(compact)), 0);
        uint64_t hi_u64 = vget_lane_u64(vreinterpret_u64_f64(vget_high_f64(compact)), 0);

        /* Non-temporal pair store: hint the CPU that these 16 bytes are
         * write-once, no later reuse, please skip the cache. */
        __asm__ __volatile__(
            "stnp %[lo], %[hi], [%[base]]"
            : /* no outputs */
            : [lo] "r"(lo_u64), [hi] "r"(hi_u64), [base] "r"(buf + cur)
            : "memory"
        );

        size_t pc = vaddvq_u64(one_bits);
        cur += pc;
    }

    /* Tail. */
    for (size_t j = chunks * 2; j < n; j++) {
        double x = input[j];
        buf[cur] = x;
        cur += (x > threshold) ? 1 : 0;
    }

    *out = buf;
    return cur;
}

/*
 * V3: same as V2 but processes 4 elements per iteration (two 2-lane blocks
 * interleaved). Halves the loop overhead.
 */
__attribute__((target("neon")))
static size_t filter_simd_neon_4lane(const double *input, size_t n, double threshold,
                                     double **out) {
    double *buf = (double *)calloc(n, sizeof(double));
    if (!buf) { *out = NULL; return 0; }
    size_t cur = 0;
    float64x2_t vth = vdupq_n_f64(threshold);
    size_t super_chunks = n / 4;

    for (size_t i = 0; i < super_chunks; i++) {
        /* Two independent 2-lane blocks. */
        float64x2_t lo1 = vld1q_f64(input + 4 * i);
        float64x2_t lo2 = vld1q_f64(input + 4 * i + 2);

        uint64x2_t m1 = vcgtq_f64(lo1, vth);
        uint64x2_t m2 = vcgtq_f64(lo2, vth);
        uint64x2_t ob1 = vshrq_n_u64(m1, 63);
        uint64x2_t ob2 = vshrq_n_u64(m2, 63);

        float64x2_t swap1 = vextq_f64(lo1, lo1, 1);
        float64x2_t swap2 = vextq_f64(lo2, lo2, 1);
        float64x2_t c1 = vbslq_f64(m1, lo1, swap1);
        float64x2_t c2 = vbslq_f64(m2, lo2, swap2);

        size_t pc1 = vaddvq_u64(ob1);
        size_t pc2 = vaddvq_u64(ob2);

        /* The two vector stores may target the same cache line; that's fine
         * — they will coalesce in the store buffer. */
        vst1q_f64(buf + cur, c1);
        vst1q_f64(buf + cur + pc1, c2);

        cur += pc1 + pc2;
    }

    /* Tail: 0..3 leftover elements after the last 4-wide chunk. */
    size_t processed = super_chunks * 4;
    for (size_t j = processed; j < n; j++) {
        double x = input[j];
        buf[cur] = x;
        cur += (x > threshold) ? 1 : 0;
    }

    *out = buf;
    return cur;
}

#endif /* HAVE_NEON */

/* ----------------------------------------------------------------------------
 * A memcpy "ceiling" — how fast can we just copy N doubles? Useful as a
 * reference: a perfect in-register SIMD filter can't go faster than this.
 * -------------------------------------------------------------------------- */
static size_t copy_ceiling(const double *input, size_t n, double **out) {
    double *buf = (double *)malloc(n * sizeof(double));
    if (!buf) { *out = NULL; return 0; }
    memcpy(buf, input, n * sizeof(double));
    *out = buf;
    return n;
}

/* ----------------------------------------------------------------------------
 * NEON variants — non-aarch64 fallback. On aarch64, the dispatcher below calls
 * the __attribute__((target("neon"))) versions directly. On other arches we
 * don't have NEON, so we just fall back to scalar branchless so the program
 * still builds and runs correctly.
 * -------------------------------------------------------------------------- */
#if !HAVE_NEON
static size_t filter_simd_neon(const double *i, size_t n, double t, double **o)
        { return filter_branchless(i, n, t, o); }
static size_t filter_simd_neon_v2(const double *i, size_t n, double t, double **o)
        { return filter_branchless(i, n, t, o); }
static size_t filter_simd_neon_v2_nozero(const double *i, size_t n, double t, double **o)
        { return filter_branchless_nozero(i, n, t, o); }
static size_t filter_simd_neon_stnp(const double *i, size_t n, double t, double **o)
        { return filter_branchless(i, n, t, o); }
static size_t filter_simd_neon_4lane(const double *i, size_t n, double t, double **o)
        { return filter_branchless(i, n, t, o); }
#endif

/* ----------------------------------------------------------------------------
 * Benchmark helpers.
 * -------------------------------------------------------------------------- */
typedef size_t (*filter_fn)(const double *input, size_t n, double threshold,
                            double **out);

static double now_ms(void) {
#if defined(__APPLE__)
    /* mach_absolute_time gives nanosecond-class resolution on macOS; clock_gettime
     * with CLOCK_MONOTONIC sometimes returns much coarser ticks here. */
    static mach_timebase_info_data_t tb = {0, 0};
    if (tb.denom == 0) mach_timebase_info(&tb);
    uint64_t t = mach_absolute_time();
    return (double)t * (double)tb.numer / (double)tb.denom / 1e6;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
#endif
}

/* Compiler barrier — the C counterpart of Rust's std::hint::black_box.
 *
 * bench_min hands the filter's output buffer straight to free() without ever
 * reading it. Since every filter is static and lives in this same translation
 * unit, clang inlines it, proves the stores are never observed, and deletes the
 * whole loop as dead. Only the variants with a side effect it can't see through
 * survived: filter_iter (realloc inside the loop), the _nozero pair (trailing
 * realloc has to preserve the contents) and stnp (asm with a "memory" clobber).
 * The rest benchmarked an empty loop and printed 0.00 ms.
 *
 * The empty asm forces `p` into a register and clobbers memory, so the compiler
 * must assume the buffer is read here and commit every prior store. Must sit
 * inside the timed region or the stores can sink past the closing now_ms(). */
static inline void black_box(const void *p) {
    __asm__ __volatile__("" : : "r"(p) : "memory");
}

/* Run `iters` repetitions of f(input, n, threshold), discard the first
 * `warmup`, and return the minimum elapsed time in milliseconds. */
static double bench_min(filter_fn f, const double *input, size_t n,
                        double threshold, size_t warmup, size_t iters) {
    double *buf = NULL;
    for (size_t i = 0; i < warmup; i++) {
        size_t len = f(input, n, threshold, &buf);
        black_box(buf);
        free(buf); buf = NULL;
        (void)len;
    }
    double best = 1e18;
    for (size_t i = 0; i < iters; i++) {
        double t0 = now_ms();
        size_t len = f(input, n, threshold, &buf);
        black_box(buf);
        double dt = now_ms() - t0;
        free(buf); buf = NULL;
        (void)len;
        if (dt < best) best = dt;
    }
    return best;
}

/* Same shape but for the no-threshold copy ceiling. */
static double bench_min_copy(const double *input, size_t n,
                             size_t warmup, size_t iters) {
    double *buf = NULL;
    for (size_t i = 0; i < warmup; i++) {
        size_t len = copy_ceiling(input, n, &buf);
        black_box(buf);
        free(buf); buf = NULL;
        (void)len;
    }
    double best = 1e18;
    for (size_t i = 0; i < iters; i++) {
        double t0 = now_ms();
        size_t len = copy_ceiling(input, n, &buf);
        black_box(buf);
        double dt = now_ms() - t0;
        free(buf); buf = NULL;
        (void)len;
        if (dt < best) best = dt;
    }
    return best;
}

/* ----------------------------------------------------------------------------
 * Output helpers.
 * -------------------------------------------------------------------------- */
static void print_row(const char *label, double t_ms) {
    printf("  %-10s  %8.2f ms\n", label, t_ms);
}

static void print_table_row(size_t kept_pct, size_t out_size,
                            double t_iter, double t_pre,
                            double t_bl, double t_bl_nozero,
                            double t_simd, double t_simd_v2,
                            double t_simd_v2_nozero,
                            double t_simd_4l, double t_stnp) {
    printf("  %3zu%%  %7zu  %6.2f   %6.2f   %6.2f   %6.2f   %6.2f   %6.2f   %6.2f   %6.2f   %6.2f\n",
           kept_pct, out_size,
           t_iter, t_pre, t_bl, t_bl_nozero,
           t_simd, t_simd_v2, t_simd_v2_nozero, t_simd_4l, t_stnp);
}

/* ----------------------------------------------------------------------------
 * Correctness check: every NEON variant must produce the same output as the
 * scalar branchless for every threshold (length match + first/middle/last
 * element match).
 * -------------------------------------------------------------------------- */
static void check_eq(const char *variant, const double *ref, size_t ref_len,
                     const double *got, size_t got_len, const char *label) {
    if (ref_len != got_len) {
        fprintf(stderr, "[%s / %s] length mismatch: scalar=%zu, simd=%zu\n",
                label, variant, ref_len, got_len);
        exit(1);
    }
    if (got_len > 0) {
        if (ref[0] != got[0]) {
            fprintf(stderr, "[%s / %s] first element %.6f vs %.6f\n",
                    label, variant, ref[0], got[0]);
            exit(1);
        }
        if (ref[ref_len / 2] != got[got_len / 2]) {
            fprintf(stderr, "[%s / %s] middle element %.6f vs %.6f\n",
                    label, variant, ref[ref_len / 2], got[got_len / 2]);
            exit(1);
        }
        if (ref[ref_len - 1] != got[got_len - 1]) {
            fprintf(stderr, "[%s / %s] last element %.6f vs %.6f\n",
                    label, variant, ref[ref_len - 1], got[got_len - 1]);
            exit(1);
        }
    }
}

/* ----------------------------------------------------------------------------
 * main
 * -------------------------------------------------------------------------- */
int main(void) {
    printf("=== Branchless C — reproduction on this machine ===\n");
    printf("Target: %s (NEON %s)\n",
#if defined(__aarch64__)
           "aarch64",
#else
           "unknown",
#endif
           HAVE_NEON ? "enabled" : "fallback");
    printf("N = %d f64 values, uniformly distributed in [0.0, 100.0)\n\n", N);

    /* ---- Generate a deterministic input once ---- */
    splitmix64 rng = { 0xDEADBEEFCAFEBABEULL };
    double *input = (double *)malloc(N * sizeof(double));
    if (!input) { fprintf(stderr, "malloc failed\n"); return 1; }
    for (size_t i = 0; i < N; i++) {
        input[i] = splitmix64_next_f64(&rng);
    }

    /* ---- Verify correctness ---- */
    struct { size_t kept_pct; const char *label; double threshold; } cases[] = {
        { 1,  "1%",  99.0 },
        { 25, "25%", 75.0 },
        { 50, "50%", 50.0 },
        { 75, "75%", 25.0 },
        { 99, "99%",  1.0 },
    };
    const size_t n_cases = sizeof(cases) / sizeof(cases[0]);

    for (size_t ci = 0; ci < n_cases; ci++) {
        double *ref = NULL;
        size_t ref_len = filter_branchless(input, N, cases[ci].threshold, &ref);

        double *got = NULL; size_t got_len;
        const struct { const char *name; filter_fn f; } variants[] = {
            { "bl_nozero",       filter_branchless_nozero    },
            { "neon",            filter_simd_neon            },
            { "neon_v2",         filter_simd_neon_v2         },
            { "neon_v2_nozero",  filter_simd_neon_v2_nozero  },
            { "neon_4lane",      filter_simd_neon_4lane      },
            { "stnp",            filter_simd_neon_stnp       },
        };
        for (size_t vi = 0; vi < sizeof(variants) / sizeof(variants[0]); vi++) {
            got_len = variants[vi].f(input, N, cases[ci].threshold, &got);
            check_eq(variants[vi].name, ref, ref_len, got, got_len, cases[ci].label);
            free(got); got = NULL;
        }
        free(ref);
    }
    printf("Correctness: NEON variants match scalar branchless for all cases.\n\n");

    /* ---- Table 1: selectivity sweep on shuffled data ---- */
    printf("Table 1 \u2014 selectivity sweep on shuffled random data\n");
    printf("  kept   out sz   iter    prealloc  branchless  bl_nozero  neon   v2    v2_nozero  4lane  stnp\n");
    printf("  -----  -------  ------  --------  ---------  ---------  ------  ----  ---------  ------  ----\n");

    const size_t warmup = 3;
    const size_t iters = 30;

    /* Store per-case results so we can emit the summary at 50% later. */
    double r_iter = 0, r_pre = 0, r_bl = 0, r_bl_nozero = 0;
    double r_simd = 0, r_simd_v2 = 0, r_simd_v2_nozero = 0;
    double r_simd_4l = 0, r_stnp = 0;

    for (size_t ci = 0; ci < n_cases; ci++) {
        double *ref = NULL;
        size_t out_size = filter_branchless(input, N, cases[ci].threshold, &ref);
        free(ref);

        double t_iter = bench_min(filter_iter,            input, N, cases[ci].threshold, warmup, iters);
        double t_pre  = bench_min(filter_prealloc,        input, N, cases[ci].threshold, warmup, iters);
        double t_bl   = bench_min(filter_branchless,      input, N, cases[ci].threshold, warmup, iters);
        double t_bl_n = bench_min(filter_branchless_nozero,input, N, cases[ci].threshold, warmup, iters);
        double t_simd = bench_min(filter_simd_neon,        input, N, cases[ci].threshold, warmup, iters);
        double t_v2   = bench_min(filter_simd_neon_v2,     input, N, cases[ci].threshold, warmup, iters);
        double t_v2n  = bench_min(filter_simd_neon_v2_nozero, input, N, cases[ci].threshold, warmup, iters);
        double t_4l   = bench_min(filter_simd_neon_4lane,  input, N, cases[ci].threshold, warmup, iters);
        double t_stnp = bench_min(filter_simd_neon_stnp,   input, N, cases[ci].threshold, warmup, iters);

        print_table_row(cases[ci].kept_pct, out_size,
                        t_iter, t_pre, t_bl, t_bl_n,
                        t_simd, t_v2, t_v2n, t_4l, t_stnp);

        if (cases[ci].kept_pct == 50) {
            r_iter          = t_iter;
            r_pre           = t_pre;
            r_bl            = t_bl;
            r_bl_nozero     = t_bl_n;
            r_simd          = t_simd;
            r_simd_v2       = t_v2;
            r_simd_v2_nozero= t_v2n;
            r_simd_4l       = t_4l;
            r_stnp          = t_stnp;
        }
    }
    printf("\n");

    /* ---- Table 2: shuffled vs sorted, same code (filter_iter), 50% kept ---- */
    printf("Table 2 \u2014 shuffled vs sorted, same code (filter_iter), 50%% kept\n");
    /* Sort a copy of the input — qsort doubles via comparator. */
    double *sorted = (double *)malloc(N * sizeof(double));
    if (!sorted) { fprintf(stderr, "malloc failed\n"); return 1; }
    memcpy(sorted, input, N * sizeof(double));
    int cmp_double(const void *, const void *);
    qsort(sorted, N, sizeof(double), cmp_double);
    double t_shuf = bench_min(filter_iter, input,  N, 50.0, warmup, iters);
    double t_sort = bench_min(filter_iter, sorted, N, 50.0, warmup, iters);
    print_row("shuffled", t_shuf);
    print_row("sorted",   t_sort);
    printf("  speedup on sorted: %.2fx\n", t_shuf / t_sort);
    printf("\n");

    /* ---- Table 3: memcpy ceiling ---- */
    printf("Table 3 \u2014 memcpy ceiling (how fast can we just copy N f64?)\n");
    double t_copy = bench_min_copy(input, N, warmup, iters);
    print_row("copy", t_copy);
    printf("\n");

    /* ---- Summary ---- */
    printf("Summary (speedup vs `filter_iter` at 50%% kept)\n");
    printf("  variant          time      vs iter   vs copy_ceil\n");
    printf("  filter_iter        %7.2f ms   1.00x     %.2fx\n", r_iter, r_iter / t_copy);
    printf("  prealloc           %7.2f ms   %.2fx     %.2fx\n", r_pre,  r_iter / r_pre,  r_pre  / t_copy);
    printf("  branchless         %7.2f ms   %.2fx     %.2fx\n", r_bl,   r_iter / r_bl,   r_bl   / t_copy);
    printf("  branchless_nozero  %7.2f ms   %.2fx     %.2fx\n", r_bl_nozero, r_iter / r_bl_nozero, r_bl_nozero / t_copy);
    printf("  neon               %7.2f ms   %.2fx     %.2fx\n", r_simd, r_iter / r_simd, r_simd / t_copy);
    printf("  neon_v2            %7.2f ms   %.2fx     %.2fx\n", r_simd_v2, r_iter / r_simd_v2, r_simd_v2 / t_copy);
    printf("  neon_v2_nozero     %7.2f ms   %.2fx     %.2fx\n", r_simd_v2_nozero, r_iter / r_simd_v2_nozero, r_simd_v2_nozero / t_copy);
    printf("  neon_4lane         %7.2f ms   %.2fx     %.2fx\n", r_simd_4l, r_iter / r_simd_4l, r_simd_4l / t_copy);
    printf("  stnp (NT pair)     %7.2f ms   %.2fx     %.2fx\n", r_stnp, r_iter / r_stnp, r_stnp / t_copy);
    printf("\n");

    free(sorted);
    free(input);
    return 0;
}

int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}