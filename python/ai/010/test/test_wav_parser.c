/*
 * test_wav_parser.c — Unit tests for the WAV parser.
 *
 * Generates a 1-second sine WAV with ffmpeg, opens it, and asserts
 * the parsed fields match what ffprobe reports.
 */

#include "wav_parser.h"
#include "util.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int test_open_basic(void) {
    const char* path = "/tmp/wav2aac_test/test_1s.wav";
    wav_t w;
    int rc = wav_open(path, &w);
    if (rc != W2A_OK) { fprintf(stderr, "open failed: %d\n", rc); return 1; }
    if (w.info.sample_rate != 48000)  { fprintf(stderr, "sr=%u\n", w.info.sample_rate); return 1; }
    if (w.info.channels    != 2)      { fprintf(stderr, "ch=%u\n", w.info.channels);    return 1; }
    if (w.info.bits_per_sample != 16) { fprintf(stderr, "bd=%u\n", w.info.bits_per_sample); return 1; }
    if (w.info.format_tag  != 1)      { fprintf(stderr, "ft=%u\n", w.info.format_tag);  return 1; }
    if (w.info.total_frames != 48000) { fprintf(stderr, "tf=%llu\n", (unsigned long long)w.info.total_frames); return 1; }
    wav_close(&w);
    return 0;
}

static int test_open_24bit(void) {
    const char* path = "/tmp/wav2aac_test/test_24b_mono.wav";
    wav_t w;
    int rc = wav_open(path, &w);
    if (rc != W2A_OK) { fprintf(stderr, "open failed: %d\n", rc); return 1; }
    if (w.info.sample_rate != 44100)  { fprintf(stderr, "sr=%u\n", w.info.sample_rate); return 1; }
    if (w.info.channels    != 1)      { fprintf(stderr, "ch=%u\n", w.info.channels);    return 1; }
    if (w.info.bits_per_sample != 24) { fprintf(stderr, "bd=%u\n", w.info.bits_per_sample); return 1; }
    wav_close(&w);
    return 0;
}

static int test_open_float(void) {
    const char* path = "/tmp/wav2aac_test/test_f32_stereo.wav";
    wav_t w;
    int rc = wav_open(path, &w);
    if (rc != W2A_OK) { fprintf(stderr, "open failed: %d\n", rc); return 1; }
    if (w.info.sample_rate != 96000)  { fprintf(stderr, "sr=%u\n", w.info.sample_rate); return 1; }
    if (w.info.channels    != 2)      { fprintf(stderr, "ch=%u\n", w.info.channels);    return 1; }
    if (w.info.bits_per_sample != 32) { fprintf(stderr, "bd=%u\n", w.info.bits_per_sample); return 1; }
    if (w.info.format_tag  != 3)      { fprintf(stderr, "ft=%u\n", w.info.format_tag);  return 1; }
    wav_close(&w);
    return 0;
}

static int test_read_pcm(void) {
    const char* path = "/tmp/wav2aac_test/test_1s.wav";
    wav_t w;
    int rc = wav_open(path, &w);
    if (rc != W2A_OK) return 1;
    int16_t buf[1024 * 2];  /* 1024 stereo frames */
    int64_t got = wav_read_pcm(&w, buf, 1024);
    if (got != 1024) { fprintf(stderr, "got=%lld (expected 1024)\n", (long long)got); wav_close(&w); return 1; }
    wav_close(&w);
    return 0;
}

int main(void) {
    int failed = 0;
    fprintf(stderr, "[TEST] test_open_basic...  ");
    if (test_open_basic())   { fprintf(stderr, "FAIL\n"); failed++; } else fprintf(stderr, "ok\n");
    fprintf(stderr, "[TEST] test_open_24bit...  ");
    if (test_open_24bit())   { fprintf(stderr, "FAIL\n"); failed++; } else fprintf(stderr, "ok\n");
    fprintf(stderr, "[TEST] test_open_float...  ");
    if (test_open_float())   { fprintf(stderr, "FAIL\n"); failed++; } else fprintf(stderr, "ok\n");
    fprintf(stderr, "[TEST] test_read_pcm...    ");
    if (test_read_pcm())     { fprintf(stderr, "FAIL\n"); failed++; } else fprintf(stderr, "ok\n");

    if (failed) {
        fprintf(stderr, "%d test(s) failed\n", failed);
        return 1;
    }
    fprintf(stderr, "All wav_parser tests passed.\n");
    return 0;
}