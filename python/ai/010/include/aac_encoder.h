#ifndef WAV2AAC_AAC_ENCODER_H
#define WAV2AAC_AAC_ENCODER_H

#include <AudioToolbox/AudioToolbox.h>
#include <stdint.h>
#include <stddef.h>

typedef struct {
    AudioStreamBasicDescription input_fmt;   /* Linear PCM 32-bit float, source rate/ch */
    AudioStreamBasicDescription output_fmt;  /* MPEG4 AAC */
    AudioConverterRef          converter;
    uint32_t                   bitrate;      /* bps, e.g. 128000 */
    uint64_t                   total_frames_in;
    uint64_t                   total_frames_out;
    /* Scratch buffer for one AAC frame from FillComplexBuffer */
    uint8_t*                   frame_buf;
    size_t                     frame_buf_cap;
    /* Pending-frame queue. FillComplexBuffer can produce multiple AAC
     * frames per call (lookahead). We keep them all here and return one
     * per aac_enc_encode() call, so the caller never loses frames. */
    uint8_t**                  pending;
    uint32_t*                  pending_sizes;
    uint32_t                   pending_count;
    uint32_t                   pending_pos;
    uint32_t                   pending_cap;
} aac_enc_t;

/*
 * Create an AAC-LC encoder for PCM float input.
 * - sample_rate/channels: source WAV params
 * - bitrate: target bps (e.g. 128000)
 */
int aac_enc_create(aac_enc_t* enc,
                   double sample_rate,
                   uint32_t channels,
                   uint32_t bitrate);

/* Free resources. */
void aac_enc_destroy(aac_enc_t* enc);

/* Print encoder manufacturer (e.g. "Apple") and name to stdout — verify HW accel. */
void aac_enc_print_info(aac_enc_t* enc);

/* Append one PCM buffer to internal state, produce zero or more AAC frames. */
int aac_enc_encode(aac_enc_t* enc,
                   const float* pcm, uint32_t pcm_frames,
                   uint8_t** out_frames, uint32_t* out_count);

/* Signal end-of-stream; flush remaining frames. */
int aac_enc_flush(aac_enc_t* enc,
                  uint8_t** out_frames, uint32_t* out_count);

#endif /* WAV2AAC_AAC_ENCODER_H */