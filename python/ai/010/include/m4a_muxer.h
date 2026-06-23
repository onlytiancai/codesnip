#ifndef WAV2AAC_M4A_MUXER_H
#define WAV2AAC_M4A_MUXER_H

#include <stdio.h>
#include <stdint.h>

typedef struct {
    FILE*      fp;
    /* Audio params */
    uint32_t   sample_rate;
    uint32_t   channels;
    uint32_t   bitrate;        /* bps */
    /* Frame table */
    uint8_t**  frames;
    uint32_t*  frame_sizes;
    uint64_t   frame_count;
    uint64_t   frame_cap;
    uint64_t   total_pcm_frames;  /* for mdhd duration */
} m4a_mux_t;

/* Open output file and write ftyp box. */
int m4a_mux_open(m4a_mux_t* m, const char* path,
                 uint32_t sample_rate, uint32_t channels, uint32_t bitrate);

/* Append one raw AAC frame (no ADTS). */
int m4a_mux_append_frame(m4a_mux_t* m, const uint8_t* data, uint32_t size);

/*
 * Write moov box and then the two-pass mdat (ftyp + moov + mdat).
 * Must be called after all frames have been appended.
 */
int m4a_mux_finalize(m4a_mux_t* m);

/* Close file and free. */
void m4a_mux_close(m4a_mux_t* m);

#endif /* WAV2AAC_M4A_MUXER_H */