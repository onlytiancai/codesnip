#ifndef WAV2AAC_BENCHMARK_H
#define WAV2AAC_BENCHMARK_H

#include <stdint.h>
#include <stdio.h>

typedef struct {
    double     wall_sec;        /* wall-clock duration */
    double     cpu_user_sec;    /* user CPU time */
    double     cpu_sys_sec;     /* system CPU time */
    uint64_t   peak_rss_bytes;  /* peak resident set size */
    uint64_t   out_bytes;       /* output file size */
    double     realtime_factor; /* wall_sec / (frames / sample_rate) */
} bench_result_t;

/* Get current monotonic time as seconds (high-resolution). */
double bench_now(void);

/* Sample RSS of current process via mach task_info. */
uint64_t bench_peak_rss(void);

/* Compute realtime factor given total PCM frames and sample rate. */
double bench_realtime(double wall_sec, uint64_t total_frames, uint32_t sample_rate);

/* Pretty-print result as one CSV row to FILE*. */
void bench_print_csv(FILE* out, const char* tool, const char* file,
                     const bench_result_t* r);

#endif /* WAV2AAC_BENCHMARK_H */