/*
 * benchmark.c — Timing and memory sampling helpers.
 */

#include "benchmark.h"
#include "util.h"
#include <mach/mach.h>
#include <mach/task_info.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <time.h>
#include <stdio.h>
#include <string.h>

double bench_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

uint64_t bench_peak_rss(void) {
    struct mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    kern_return_t kr = task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                                  (task_info_t)&info, &count);
    if (kr != KERN_SUCCESS) return 0;
    return (uint64_t)info.resident_size;
}

double bench_realtime(double wall_sec, uint64_t total_frames, uint32_t sample_rate) {
    if (sample_rate == 0) return 0.0;
    double audio_sec = (double)total_frames / (double)sample_rate;
    if (audio_sec <= 0.0) return 0.0;
    return wall_sec / audio_sec;
}

void bench_print_csv(FILE* out, const char* tool, const char* file,
                     const bench_result_t* r) {
    fprintf(out, "%s,%s,%.4f,%.4f,%.4f,%llu,%llu,%.4f\n",
            tool, file, r->wall_sec, r->cpu_user_sec, r->cpu_sys_sec,
            (unsigned long long)r->peak_rss_bytes,
            (unsigned long long)r->out_bytes,
            r->realtime_factor);
}