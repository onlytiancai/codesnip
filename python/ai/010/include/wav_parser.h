#ifndef WAV2AAC_WAV_PARSER_H
#define WAV2AAC_WAV_PARSER_H

#include <stdio.h>
#include <stdint.h>

typedef struct {
    /* RIFF/WAV metadata */
    uint16_t format_tag;        /* 1 = PCM, 3 = IEEE float */
    uint16_t channels;
    uint32_t sample_rate;
    uint32_t avg_bytes_per_sec;
    uint16_t block_align;
    uint16_t bits_per_sample;

    /* Computed */
    uint64_t total_frames;      /* number of sample frames in data chunk */
    uint64_t data_size;         /* raw data chunk size in bytes */
    int64_t  data_offset;       /* file offset of first data byte */
} wav_info_t;

typedef struct {
    FILE*     fp;
    wav_info_t info;
    uint32_t  downsample;   /* 1 = none, 2 = skip every other, etc. */
} wav_t;

/* Open and parse WAV header. Returns W2A_OK on success. */
int wav_open(const char* path, wav_t* out);

/* Close file handle. */
void wav_close(wav_t* w);

/*
 * Read PCM frames from data chunk.
 * `dst`     : caller-provided buffer, must hold frames * block_align bytes.
 * `frames`  : max frames to read.
 * Returns number of frames actually read (0 at EOF), or negative W2A_ERR_*.
 */
int64_t wav_read_pcm(wav_t* w, void* dst, uint64_t frames);

/*
 * Configure a frame downsample step. When set to N>1, the reader will
 * advance N source frames per consumed frame (skip-and-take). Used when
 * the caller wants to feed a higher-rate WAV into a 48kHz encoder. */
void wav_set_downsample(wav_t* w, uint32_t factor);
uint32_t wav_downsample(const wav_t* w);

/* Seek to absolute frame position from start of data. */
int wav_seek_frame(wav_t* w, uint64_t frame);

#endif /* WAV2AAC_WAV_PARSER_H */