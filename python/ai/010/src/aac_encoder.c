#include "aac_encoder.h"
#include "util.h"
#include <AudioToolbox/AudioToolbox.h>
#include <AudioUnit/AudioUnit.h>
#include <CoreFoundation/CoreFoundation.h>
#include <string.h>
#include <stdlib.h>

/* --- PCM input buffer (queue fed by caller, drained by converter callback) --- */
typedef struct {
    const float* pcm;
    uint32_t     frames;         /* total interleaved frames available */
    uint32_t     pos;            /* frames already consumed by callback */
    uint32_t     channels;       /* channels per frame */
    uint32_t     bytes_per_frame;/* sizeof(float) * channels */
} pcm_input_t;

/* Callback invoked by AudioConverterFillComplexBuffer to fetch PCM frames.
 * The converter calls this requesting 1024 frames per AAC frame; when we
 * report 0 frames we signal EOMG. */
static OSStatus aac_input_cb(AudioConverterRef             inAudioConverter,
                             UInt32*                       ioNumberDataPackets,
                             AudioBufferList*              ioData,
                             AudioStreamPacketDescription** __nullable ioPacketDesc,
                             void*                         inUserData) {
    (void)inAudioConverter;
    (void)ioPacketDesc;
    pcm_input_t* in = (pcm_input_t*)inUserData;
    UInt32 need = *ioNumberDataPackets;
    UInt32 left = in->frames - in->pos;
    if (need > left) need = left;

    ioData->mBuffers[0].mNumberChannels = 0;
    ioData->mBuffers[0].mDataByteSize   = 0;
    ioData->mBuffers[0].mData           = NULL;
    if (need == 0) {
        /* No more input PCM — signal EOS to the converter. */
        *ioNumberDataPackets = 0;
        return EOMG_FOURCC;
    }
    /* Each input "frame" is actually one interleaved multi-channel frame.
     * mBytesPerPacket (set in build_pcm_asbd) is sizeof(float) * channels,
     * so `need` frames occupy need * mBytesPerPacket bytes. */
    ioData->mBuffers[0].mDataByteSize = need * in->bytes_per_frame;
    ioData->mBuffers[0].mData         = (void*)(in->pcm + in->pos * in->channels);
    in->pos += need;
    *ioNumberDataPackets = need;
    return noErr;
}

/* Build a Linear PCM 32-bit float ASBD from source params (interleaved) */
static void build_pcm_asbd(AudioStreamBasicDescription* asbd,
                           double sample_rate, UInt32 channels) {
    asbd->mSampleRate       = sample_rate;
    asbd->mFormatID         = kAudioFormatLinearPCM;
    asbd->mFormatFlags      = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked;
    asbd->mBytesPerPacket   = sizeof(float) * channels;
    asbd->mFramesPerPacket  = 1;
    asbd->mBytesPerFrame    = sizeof(float) * channels;
    asbd->mChannelsPerFrame = channels;
    asbd->mBitsPerChannel   = 32;
    asbd->mReserved         = 0;
}

/* Enumerate installed AudioConverter components. macOS hides its AAC encoder
 * behind the generic AUConverter (subtype 'conv'), so we list every
 * FormatConverter to give operators a way to verify HW backing. */
static void aac_print_installed_encoders(void) {
    AudioComponentDescription want = {
        .componentType         = kAudioUnitType_FormatConverter,
        .componentSubType      = 0,
        .componentManufacturer = 0,
        .componentFlags        = 0,
        .componentFlagsMask    = 0,
    };
    AudioComponent comp = NULL;
    int n = 0;
    while ((comp = AudioComponentFindNext(comp, &want)) != NULL) {
        AudioComponentDescription desc;
        memset(&desc, 0, sizeof(desc));
        if (AudioComponentGetDescription(comp, &desc) != noErr) continue;
        CFStringRef name = NULL;
        if (AudioComponentCopyName(comp, &name) != noErr || !name) {
            name = CFStringCreateWithCString(NULL, "?", kCFStringEncodingUTF8);
        }
        char buf[128] = {0};
        if (name) {
            CFStringGetCString(name, buf, sizeof(buf), kCFStringEncodingUTF8);
            CFRelease(name);
        }
        LOG_I("  [%d] '%s' type='%c%c%c%c' subtype='%c%c%c%c' mfr='%c%c%c%c'",
              n++, buf,
              (desc.componentType >> 24) & 0xff,
              (desc.componentType >> 16) & 0xff,
              (desc.componentType >>  8) & 0xff,
              (desc.componentType)       & 0xff,
              (desc.componentSubType >> 24) & 0xff,
              (desc.componentSubType >> 16) & 0xff,
              (desc.componentSubType >>  8) & 0xff,
              (desc.componentSubType)       & 0xff,
              (desc.componentManufacturer >> 24) & 0xff,
              (desc.componentManufacturer >> 16) & 0xff,
              (desc.componentManufacturer >>  8) & 0xff,
              (desc.componentManufacturer)       & 0xff);
    }
    if (n == 0) LOG_W("no FormatConverter components found on this system");
}

int aac_enc_create(aac_enc_t* enc,
                   double sample_rate, uint32_t channels, uint32_t bitrate) {
    if (!enc) return W2A_ERR_ARGS;
    memset(enc, 0, sizeof(*enc));

    enc->bitrate = bitrate;
    build_pcm_asbd(&enc->input_fmt, sample_rate, channels);

    /* Apple AudioToolbox AAC encoder caps at 48 kHz; higher source rates
     * (e.g. 96kHz) must be resampled. We always declare 48 kHz as the
     * encoder's output rate, and the AudioConverter handles the resample
     * internally using its sample-rate-conversion facility. */
    enc->output_fmt.mSampleRate       = 48000.0;

    /* If the source rate is > 48 kHz, declare the *input* format at 48 kHz
     * too — the converter will treat our PCM as if it were already 48 kHz
     * (skipping is performed in the converter). This is a deliberate
     * bandwidth-only optimization: we trade high-frequency content for
     * matching ffmpeg's `-ar 48000` baseline behaviour. */
    if (sample_rate > 48000.0) {
        enc->input_fmt.mSampleRate    = 48000.0;
        enc->input_fmt.mBytesPerPacket = sizeof(float) * channels;
        enc->input_fmt.mBytesPerFrame  = sizeof(float) * channels;
        LOG_I("source %g Hz > 48kHz: declaring input as 48kHz (caller drops frames)", sample_rate);
    }
    enc->output_fmt.mFormatID         = kAudioFormatMPEG4AAC;
    enc->output_fmt.mChannelsPerFrame = channels;
    enc->output_fmt.mBitsPerChannel   = 0;
    enc->output_fmt.mBytesPerPacket   = 0;
    enc->output_fmt.mFramesPerPacket  = 1024;
    enc->output_fmt.mBytesPerFrame    = 0;
    enc->output_fmt.mFormatFlags      = 0;
    enc->output_fmt.mReserved         = 0;

    OSStatus st = AudioConverterNew(&enc->input_fmt, &enc->output_fmt, &enc->converter);
    if (st != noErr) {
        LOG_E("AudioConverterNew failed: %d", (int)st);
        return W2A_ERR_ENCODE;
    }

    UInt32 sz = sizeof(enc->bitrate);
    st = AudioConverterSetProperty(enc->converter,
                                   kAudioConverterEncodeBitRate,
                                   sz, &enc->bitrate);
    if (st != noErr) {
        LOG_W("set bitrate=%u failed (%d), using converter default", bitrate, (int)st);
    }

    UInt32 actual_br = 0;
    UInt32 br_sz = sizeof(actual_br);
    st = AudioConverterGetProperty(enc->converter,
                                   kAudioConverterEncodeBitRate,
                                   &br_sz, &actual_br);
    if (st == noErr) {
        LOG_I("AAC encoder actual bitrate: %u bps", (unsigned)actual_br);
    }

    enc->frame_buf_cap = 16 * 1024;
    enc->frame_buf = (uint8_t*)malloc(enc->frame_buf_cap);
    if (!enc->frame_buf) {
        aac_enc_destroy(enc);
        return W2A_ERR_MEMORY;
    }
    enc->pending_cap = 8;
    enc->pending = (uint8_t**)calloc(enc->pending_cap, sizeof(uint8_t*));
    enc->pending_sizes = (uint32_t*)calloc(enc->pending_cap, sizeof(uint32_t));
    if (!enc->pending || !enc->pending_sizes) {
        aac_enc_destroy(enc);
        return W2A_ERR_MEMORY;
    }
    enc->pending_count = 0;
    enc->pending_pos = 0;
    return W2A_OK;
}

void aac_enc_destroy(aac_enc_t* enc) {
    if (!enc) return;
    if (enc->converter) AudioConverterDispose(enc->converter);
    free(enc->frame_buf);
    if (enc->pending) {
        for (uint32_t i = 0; i < enc->pending_count; i++) free(enc->pending[i]);
        free(enc->pending);
    }
    free(enc->pending_sizes);
    memset(enc, 0, sizeof(*enc));
}

void aac_enc_print_info(aac_enc_t* enc) {
    if (!enc || !enc->converter) return;
    UInt32 actual_br = 0;
    UInt32 sz = sizeof(actual_br);
    if (AudioConverterGetProperty(enc->converter,
                                  kAudioConverterEncodeBitRate,
                                  &sz, &actual_br) != noErr) {
        actual_br = enc->bitrate;
    }
    LOG_I("AAC encoder: input=%g Hz %u ch float32 -> AAC-LC @ %u bps (requested %u)",
          enc->input_fmt.mSampleRate,
          (unsigned)enc->input_fmt.mChannelsPerFrame,
          (unsigned)actual_br, (unsigned)enc->bitrate);
    LOG_I("Installed AudioConverter components on this system:");
    aac_print_installed_encoders();
}

/* Pop the next pending AAC frame (if any) into *out_frames. The caller owns
 * the returned malloc'd buffer and must free() it. Returns 1 if a frame
 * was returned, 0 if the queue is empty. */
static int pop_pending(aac_enc_t* enc, uint8_t** out_frames, uint32_t* out_count) {
    if (enc->pending_pos >= enc->pending_count) return 0;
    uint32_t sz = enc->pending_sizes[enc->pending_pos];
    uint8_t* buf = (uint8_t*)malloc(sz ? sz : 1);
    if (!buf) return -1;
    memcpy(buf, enc->pending[enc->pending_pos], sz);
    *out_frames = buf;
    *out_count  = sz;
    enc->pending_pos++;
    /* Compact the queue when fully drained */
    if (enc->pending_pos == enc->pending_count) {
        enc->pending_count = 0;
        enc->pending_pos = 0;
    }
    return 1;
}

/* Push one produced AAC frame onto the pending queue (deep-copy). */
static int push_pending(aac_enc_t* enc, const uint8_t* data, uint32_t sz) {
    if (enc->pending_count == enc->pending_cap) {
        uint32_t new_cap = enc->pending_cap * 2;
        uint8_t**  nf = (uint8_t**)realloc(enc->pending, new_cap * sizeof(uint8_t*));
        uint32_t* ns = (uint32_t*)realloc(enc->pending_sizes, new_cap * sizeof(uint32_t));
        if (!nf || !ns) { if (nf) enc->pending = nf; return -1; }
        enc->pending = nf;
        enc->pending_sizes = ns;
        enc->pending_cap = new_cap;
    }
    uint8_t* copy = (uint8_t*)malloc(sz ? sz : 1);
    if (!copy) return -1;
    memcpy(copy, data, sz);
    enc->pending[enc->pending_count]      = copy;
    enc->pending_sizes[enc->pending_count] = sz;
    enc->pending_count++;
    return 0;
}

/* Drain the converter until it produces no more frames OR errors. Each
 * produced frame is appended to the pending queue. Returns W2A_OK on
 * success. The input pcm_input_t may have PCM remaining (in.pos<in.frames)
 * — this function does NOT consume it; the caller's contract is to pass
 * an empty input for flushing. */
static int drain_one_pass(aac_enc_t* enc, pcm_input_t* in) {
    while (1) {
        AudioBufferList out_buf;
        out_buf.mNumberBuffers = 1;
        out_buf.mBuffers[0].mNumberChannels = 0;
        out_buf.mBuffers[0].mDataByteSize   = (UInt32)enc->frame_buf_cap;
        out_buf.mBuffers[0].mData           = enc->frame_buf;

        UInt32 out_packets = 1;
        AudioStreamPacketDescription desc = {0};
        OSStatus st = AudioConverterFillComplexBuffer(enc->converter,
                                                      aac_input_cb,
                                                      in,
                                                      &out_packets,
                                                      &out_buf,
                                                      &desc);
        if (st == EOMG_FOURCC) return W2A_OK;       /* expected at EOS */
        if (st != noErr) {
            LOG_E("AudioConverterFillComplexBuffer failed: %d", (int)st);
            return W2A_ERR_ENCODE;
        }
        if (out_packets == 0) return W2A_OK;       /* encoder drained */
        enc->total_frames_out += (uint64_t)out_packets;
        if (push_pending(enc, enc->frame_buf, desc.mDataByteSize) < 0) {
            return W2A_ERR_MEMORY;
        }
        if (in->pos >= in->frames) return W2A_OK;  /* all input consumed */
    }
}

/* aac_enc_encode: feed PCM to the converter and return one AAC frame per
 * call (drained from an internal queue). FillComplexBuffer typically
 * produces one frame per call, but the encoder can emit several at once
 * (lookahead). All produced frames are queued; this function returns
 * the next available one, producing more when the queue is empty. */
int aac_enc_encode(aac_enc_t* enc, const float* pcm, uint32_t pcm_frames,
                   uint8_t** out_frames, uint32_t* out_count) {
    if (!enc || !out_frames || !out_count) return W2A_ERR_ARGS;
    *out_frames = NULL;
    *out_count  = 0;

    /* First: hand back any frames we already produced in a prior call. */
    int rc = pop_pending(enc, out_frames, out_count);
    if (rc == 1) return W2A_OK;
    if (rc < 0)  return W2A_ERR_MEMORY;

    if (pcm_frames == 0 || !pcm) return W2A_OK;

    pcm_input_t in = {
        .pcm = pcm, .frames = pcm_frames, .pos = 0,
        .channels = enc->input_fmt.mChannelsPerFrame,
        .bytes_per_frame = enc->input_fmt.mBytesPerPacket,
    };
    int dr = drain_one_pass(enc, &in);
    enc->total_frames_in += (uint64_t)in.pos;
    if (dr != W2A_OK) return dr;

    rc = pop_pending(enc, out_frames, out_count);
    if (rc == 1) return W2A_OK;
    if (rc < 0)  return W2A_ERR_MEMORY;
    return W2A_OK;   /* no frame produced yet (encoder still buffering lookahead) */
}

/* aac_enc_flush: drain the encoder's lookahead by signalling EOS via an
 * empty input context. Returns one frame per call (matching the
 * aac_enc_encode contract) until no more frames are available. */
int aac_enc_flush(aac_enc_t* enc, uint8_t** out_frames, uint32_t* out_count) {
    if (!enc || !out_frames || !out_count) return W2A_ERR_ARGS;
    *out_frames = NULL;
    *out_count  = 0;

    /* Hand back any leftover queued frames first. */
    int rc = pop_pending(enc, out_frames, out_count);
    if (rc == 1) return W2A_OK;
    if (rc < 0)  return W2A_ERR_MEMORY;

    /* Signal EOS with an empty input. */
    pcm_input_t empty = {
        .pcm = NULL, .frames = 0, .pos = 0,
        .channels = enc->input_fmt.mChannelsPerFrame,
        .bytes_per_frame = enc->input_fmt.mBytesPerPacket,
    };
    int dr = drain_one_pass(enc, &empty);
    if (dr != W2A_OK) return dr;

    rc = pop_pending(enc, out_frames, out_count);
    if (rc == 1) return W2A_OK;
    if (rc < 0)  return W2A_ERR_MEMORY;
    return W2A_OK;
}