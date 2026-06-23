/*
 * converter.c — Top-level WAV → M4A pipeline.
 *
 * Flow:
 *   1. Parse WAV header to obtain sample_rate / channels / bit depth.
 *   2. Convert source PCM to 32-bit float (interleaved) in a scratch buffer.
 *   3. Pump float PCM into AudioConverter (AAC-LC) one FillComplexBuffer call
 *      per AAC frame; each call emits a raw AAC frame to the M4A muxer.
 *   4. Finalize the M4A container (writes ftyp + moov + mdat).
 */

#include "converter.h"
#include "wav_parser.h"
#include "aac_encoder.h"
#include "m4a_muxer.h"
#include "util.h"

#include <dispatch/dispatch.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Convert interleaved int16 → float in-place in `dst`. */
static int pcm16_to_float(const int16_t* src, float* dst, uint64_t frames, uint32_t channels) {
    for (uint64_t i = 0; i < frames * channels; i++) {
        dst[i] = (float)src[i] / 32768.0f;
    }
    return 0;
}

/* Convert interleaved int24 (24-bit little-endian) → float. */
static int pcm24_to_float(const uint8_t* src, float* dst, uint64_t frames, uint32_t channels) {
    for (uint64_t i = 0; i < frames * channels; i++) {
        const uint8_t* p = src + i * 3;
        int32_t s = (int32_t)((uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16));
        if (s & 0x800000) s |= 0xff000000;   /* sign-extend */
        dst[i] = (float)s / 8388608.0f;
    }
    return 0;
}

/* Convert interleaved int32 → float. */
static int pcm32_to_float(const int32_t* src, float* dst, uint64_t frames, uint32_t channels) {
    for (uint64_t i = 0; i < frames * channels; i++) {
        dst[i] = (float)((double)src[i] / 2147483648.0);
    }
    return 0;
}

/* Already-float pass-through. */
static int float_passthrough(const float* src, float* dst, uint64_t frames, uint32_t channels) {
    memcpy(dst, src, frames * channels * sizeof(float));
    return 0;
}

int convert_wav_to_m4a(const char* in_path, const char* out_path, uint32_t bitrate) {
    if (!in_path || !out_path) return W2A_ERR_ARGS;

    wav_t w;
    int rc = wav_open(in_path, &w);
    if (rc != W2A_OK) return rc;

    /* If source rate > 48 kHz, downsample on read so the encoder can be
     * declared at 48 kHz (the AudioToolbox AAC encoder cap). */
    uint32_t effective_sr = w.info.sample_rate;
    if (w.info.sample_rate > 48000) {
        uint32_t factor = w.info.sample_rate / 48000;
        wav_set_downsample(&w, factor);
        effective_sr = 48000;
        LOG_I("source %u Hz: downsampling by %u for AAC encoder cap",
              w.info.sample_rate, factor);
    }

    /* Setup AAC encoder (always outputs at 48 kHz; AudioConverter resamples
     * internally if the source is at a different rate). */
    aac_enc_t enc;
    rc = aac_enc_create(&enc, (double)effective_sr, w.info.channels, bitrate);
    if (rc != W2A_OK) { wav_close(&w); return rc; }
    aac_enc_print_info(&enc);

    /* The encoded AAC stream is at 48 kHz regardless of source rate
     * (AudioConverter resamples to 48 k internally). Use 48 k for the
     * muxer timescale so mdhd/esds sample-rate fields are correct, and
     * the AAC prime is expressed in 48 kHz frames for any elst edit
     * list. */
    m4a_mux_t mux;
    rc = m4a_mux_open(&mux, out_path, 48000, w.info.channels, bitrate);
    if (rc != W2A_OK) { aac_enc_destroy(&enc); wav_close(&w); return rc; }

    /* IMPORTANT: feed the entire PCM stream to the encoder in a single
     * call. AudioConverterFillComplexBuffer is stateful (it has an
     * internal resampler + encoder lookahead), and breaking the input
     * into chunks while advancing pcm_pos by 1024 (the AAC frame size)
     * corrupts that state because 44100→48000 resampling consumes ~941
     * input frames per output frame, not exactly 1024. The result is
     * silent sample loss (~0.9s for an 11s/44.1k stereo file). */
    const uint32_t CHUNK_FRAMES = 4096;
    size_t pcm_bytes = (size_t)CHUNK_FRAMES * w.info.channels * sizeof(float);
    float* pcm_chunk = (float*)malloc(pcm_bytes);
    if (!pcm_chunk) { m4a_mux_close(&mux); aac_enc_destroy(&enc); wav_close(&w); return W2A_ERR_MEMORY; }

    size_t src_bytes = (size_t)CHUNK_FRAMES * w.info.block_align;
    void* src = malloc(src_bytes);
    if (!src) { free(pcm_chunk); m4a_mux_close(&mux); aac_enc_destroy(&enc); wav_close(&w); return W2A_ERR_MEMORY; }

    /* Accumulate the entire float PCM stream into a single growing buffer
     * so the encoder sees one continuous input. */
    float*  pcm_total     = NULL;
    uint64_t pcm_total_frames = 0;
    uint64_t pcm_total_cap    = 0;

    while (1) {
        int64_t got = wav_read_pcm(&w, src, CHUNK_FRAMES);
        if (got < 0) { rc = (int)got; break; }
        if (got == 0) break;

        /* Convert to float */
        switch (w.info.format_tag) {
            case 1: /* PCM */
                switch (w.info.bits_per_sample) {
                    case 16: pcm16_to_float((const int16_t*)src, pcm_chunk, (uint64_t)got, w.info.channels); break;
                    case 24: pcm24_to_float((const uint8_t*)src, pcm_chunk, (uint64_t)got, w.info.channels); break;
                    case 32: pcm32_to_float((const int32_t*)src, pcm_chunk, (uint64_t)got, w.info.channels); break;
                    default: rc = W2A_ERR_FORMAT; goto done;
                }
                break;
            case 3: /* IEEE float */
                float_passthrough((const float*)src, pcm_chunk, (uint64_t)got, w.info.channels);
                break;
            default: rc = W2A_ERR_FORMAT; goto done;
        }

        if (pcm_total_frames + (uint64_t)got > pcm_total_cap) {
            uint64_t new_cap = pcm_total_cap ? pcm_total_cap * 2 : (uint64_t)CHUNK_FRAMES;
            while (new_cap < pcm_total_frames + (uint64_t)got) new_cap *= 2;
            float* nb = (float*)realloc(pcm_total, new_cap * w.info.channels * sizeof(float));
            if (!nb) { rc = W2A_ERR_MEMORY; goto done; }
            pcm_total     = nb;
            pcm_total_cap = new_cap;
        }
        memcpy(pcm_total + pcm_total_frames * w.info.channels,
               pcm_chunk, (size_t)got * w.info.channels * sizeof(float));
        pcm_total_frames += (uint64_t)got;
    }

    if (rc != W2A_OK) goto done;

    /* Feed the entire stream to the encoder in one call. drain_one_pass
     * inside aac_enc_encode will keep calling FillComplexBuffer until the
     * input is exhausted, queuing all produced frames. */
    {
        uint8_t* frame = NULL;
        uint32_t frame_size = 0;
        uint32_t total_in = (uint32_t)pcm_total_frames;   /* AAC frame count is < 2^32 */
        rc = aac_enc_encode(&enc, pcm_total, total_in, &frame, &frame_size);
        while (rc == W2A_OK && frame_size > 0) {
            rc = m4a_mux_append_frame(&mux, frame, frame_size);
            free(frame);
            frame = NULL; frame_size = 0;
            if (rc != W2A_OK) goto done;
            rc = aac_enc_encode(&enc, NULL, 0, &frame, &frame_size);
        }
    }

    /* Drain any residual frames from the converter (encoder lookahead). */
    {
        uint8_t* frame = NULL;
        uint32_t frame_size = 0;
        int drain_runs = 0;
        while (drain_runs < 16) {
            rc = aac_enc_flush(&enc, &frame, &frame_size);
            if (rc != W2A_OK) break;
            if (frame_size == 0) break;
            rc = m4a_mux_append_frame(&mux, frame, frame_size);
            free(frame);
            if (rc != W2A_OK) goto done;
            drain_runs++;
        }
    }

done:
    if (rc == W2A_OK) {
        /* mux.frame_count * 1024 gives the exact number of 48 kHz output
         * samples. Using this (instead of source PCM count) keeps mdhd in
         * sync with stts and avoids ffprobe's "wrong sample count"
         * warning. */
        mux.total_pcm_frames = mux.frame_count * 1024ULL;
        rc = m4a_mux_finalize(&mux);
    } else {
        LOG_E("conversion failed: %s", w2a_strerror(rc));
    }
    free(pcm_chunk);
    free(src);
    free(pcm_total);
    m4a_mux_close(&mux);
    aac_enc_destroy(&enc);
    wav_close(&w);
    return rc;
}

int convert_wav_batch(const char** paths, const char** out_paths,
                      size_t count, uint32_t bitrate) {
    if (!paths || !out_paths || count == 0) return W2A_ERR_ARGS;
    /* Use a serial queue so the per-thread setjmp/lookup is uncontended, then
     * dispatch_apply with USER_INITIATED QOS — each file is independent. */
    __block int first_err = W2A_OK;
    dispatch_queue_t q = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);
    dispatch_apply((size_t)count, q, ^(size_t i) {
        int r = convert_wav_to_m4a(paths[i], out_paths[i], bitrate);
        if (r != W2A_OK) {
            LOG_E("batch: %s -> %s failed: %s",
                  paths[i], out_paths[i], w2a_strerror(r));
            /* first_err is captured by __block, but dispatch_apply doesn't
             * guarantee any serialization between blocks. Use an atomic-ish
             * approach via dispatch_once or just use a serial counter. */
            __atomic_store_n(&first_err, r, __ATOMIC_RELAXED);
        }
    });
    return first_err;
}