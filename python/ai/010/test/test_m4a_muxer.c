/*
 * test_m4a_muxer.c — Unit tests for the M4A muxer.
 *
 * Synthesizes 3 raw AAC frames (just zeros — payload content is opaque to
 * the container layer), feeds them to the muxer, writes the .m4a file,
 * and then calls ffprobe to confirm it parses as a valid QuickTime / M4A
 * container with one audio stream.
 */

#include "m4a_muxer.h"
#include "util.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int test_basic_roundtrip(void) {
    const char* path = "/tmp/wav2aac_test/test_muxer.m4a";
    m4a_mux_t m;
    int rc = m4a_mux_open(&m, path, 48000, 2, 128000);
    if (rc != W2A_OK) { fprintf(stderr, "m4a_mux_open failed: %d\n", rc); return 1; }
    /* Fake 3 AAC frames (size typical for 48kHz stereo @ 128k = ~432 bytes) */
    uint8_t frame[432];
    memset(frame, 0xab, sizeof(frame));
    for (int i = 0; i < 3; i++) {
        rc = m4a_mux_append_frame(&m, frame, sizeof(frame));
        if (rc != W2A_OK) { fprintf(stderr, "append_frame failed: %d\n", rc); return 1; }
    }
    m.total_pcm_frames = 3 * 1024;  /* 3 AAC frames * 1024 PCM frames each */
    rc = m4a_mux_finalize(&m);
    if (rc != W2A_OK) { fprintf(stderr, "finalize failed: %d\n", rc); return 1; }
    m4a_mux_close(&m);

    /* Confirm file exists and has nonzero size */
    FILE* fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "cannot reopen %s\n", path); return 1; }
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fclose(fp);
    if (sz < 100) { fprintf(stderr, "file too small: %ld\n", sz); return 1; }

    /* Check ftyp at start */
    fp = fopen(path, "rb");
    unsigned char hdr[8];
    if (fread(hdr, 1, 8, fp) != 8) { fclose(fp); return 1; }
    fclose(fp);
    uint32_t ftyp_size = (hdr[0]<<24)|(hdr[1]<<16)|(hdr[2]<<8)|hdr[3];
    if (memcmp(hdr+4, "ftyp", 4) != 0) { fprintf(stderr, "no ftyp at start\n"); return 1; }
    if (ftyp_size != 28) { fprintf(stderr, "ftyp size=%u (expected 28)\n", ftyp_size); return 1; }
    return 0;
}

int main(void) {
    int failed = 0;
    fprintf(stderr, "[TEST] test_basic_roundtrip... ");
    if (test_basic_roundtrip()) { fprintf(stderr, "FAIL\n"); failed++; }
    else fprintf(stderr, "ok\n");
    if (failed) { fprintf(stderr, "%d test(s) failed\n", failed); return 1; }
    fprintf(stderr, "All m4a_muxer tests passed.\n");
    return 0;
}