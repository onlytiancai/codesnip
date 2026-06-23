#include "wav_parser.h"
#include "util.h"
#include <arpa/inet.h>  /* htons / ntohs */
#include <errno.h>

/* Read 4 bytes as a FourCC-style ASCII tag (big-endian on disk). */
static int read_fourcc(FILE* fp, char cc[4]) {
    return fread(cc, 1, 4, fp) == 4 ? 0 : -1;
}

static uint16_t read_u16le(FILE* fp) {
    uint8_t b[2];
    if (fread(b, 1, 2, fp) != 2) return 0;
    return (uint16_t)b[0] | ((uint16_t)b[1] << 8);
}

static uint32_t read_u32le(FILE* fp) {
    uint8_t b[4];
    if (fread(b, 1, 4, fp) != 4) return 0;
    return (uint32_t)b[0]
         | ((uint32_t)b[1] << 8)
         | ((uint32_t)b[2] << 16)
         | ((uint32_t)b[3] << 24);
}

int wav_open(const char* path, wav_t* out) {
    if (!path || !out) return W2A_ERR_ARGS;
    memset(out, 0, sizeof(*out));

    FILE* fp = fopen(path, "rb");
    if (!fp) {
        LOG_E("fopen(%s) failed: %s", path, strerror(errno));
        return W2A_ERR_OPEN;
    }
    out->fp = fp;

    /* --- RIFF header --- */
    char tag[4];
    if (read_fourcc(fp, tag) != 0 || memcmp(tag, "RIFF", 4) != 0) {
        LOG_E("missing RIFF header in %s", path);
        return W2A_ERR_FORMAT;
    }
    uint32_t riff_size = read_u32le(fp);
    (void)riff_size;  /* may be 0xFFFFFFFF for RF64; not handled */
    if (read_fourcc(fp, tag) != 0 || memcmp(tag, "WAVE", 4) != 0) {
        LOG_E("missing WAVE tag in %s", path);
        return W2A_ERR_FORMAT;
    }

    /* --- Walk chunks until we hit 'data' --- */
    int fmt_found = 0, data_found = 0;
    while (!(fmt_found && data_found)) {
        char ckid[4];
        if (read_fourcc(fp, ckid) != 0) {
            LOG_E("unexpected EOF before data chunk in %s", path);
            return W2A_ERR_FORMAT;
        }
        uint32_t cksize = read_u32le(fp);

        if (memcmp(ckid, "fmt ", 4) == 0) {
            if (cksize < 16) {
                LOG_E("fmt chunk too small (%u bytes)", cksize);
                return W2A_ERR_FORMAT;
            }
            out->info.format_tag       = read_u16le(fp);
            out->info.channels         = read_u16le(fp);
            out->info.sample_rate      = read_u32le(fp);
            out->info.avg_bytes_per_sec= read_u32le(fp);
            out->info.block_align      = read_u16le(fp);
            out->info.bits_per_sample  = read_u16le(fp);

            if (out->info.format_tag == 0xFFFE) {
                /* WAVE_FORMAT_EXTENSIBLE: read the sub-format GUID to recover
                 * the real format tag (PCM=1, FLOAT=3). */
                uint16_t cb_size = read_u16le(fp);
                if (cb_size < 22) {
                    LOG_E("WAVE_FORMAT_EXTENSIBLE: cb_size too small (%u)", cb_size);
                    return W2A_ERR_FORMAT;
                }
                (void)read_u16le(fp);  /* valid_bits_per_sample */
                (void)read_u32le(fp);  /* channel_mask */
                uint8_t guid[16];
                if (fread(guid, 1, 16, fp) != 16) return W2A_ERR_FORMAT;
                /* GUID is little-endian on disk; first 4 bytes are the format tag. */
                out->info.format_tag = (uint16_t)guid[0] | ((uint16_t)guid[1] << 8);
                /* Skip any further extension bytes */
                uint32_t consumed = 16 + 2 + 2 + 4 + 16; /* cb_size+ext+GUID */
                uint32_t skip = cksize - consumed;
                if (skip > 0 && fseek(fp, (long)skip, SEEK_CUR) != 0) {
                    return W2A_ERR_FORMAT;
                }
            } else {
                /* Plain WAVEFORMATEX (or PCM) */
                uint32_t skip = cksize - 16;
                if (skip > 0 && fseek(fp, (long)skip, SEEK_CUR) != 0) {
                    LOG_E("seek past fmt ext failed");
                    return W2A_ERR_FORMAT;
                }
            }
            fmt_found = 1;
        } else if (memcmp(ckid, "data", 4) == 0) {
            out->info.data_size   = cksize;
            out->info.data_offset = (int64_t)ftell(fp);
            if (out->info.block_align == 0) {
                LOG_E("data chunk encountered before fmt chunk set block_align");
                return W2A_ERR_FORMAT;
            }
            out->info.total_frames = cksize / out->info.block_align;
            data_found = 1;
        } else {
            /* Unknown chunk — skip its payload (and pad byte if cksize is odd) */
            LOG_D("skipping unknown chunk '%c%c%c%c' size=%u",
                  ckid[0], ckid[1], ckid[2], ckid[3], cksize);
            if (fseek(fp, (long)cksize + (cksize & 1), SEEK_CUR) != 0) {
                LOG_E("seek past unknown chunk failed");
                return W2A_ERR_FORMAT;
            }
        }
    }

    /* Validate supported formats */
    if (out->info.format_tag != 1 && out->info.format_tag != 3) {
        LOG_E("unsupported format_tag=%u (need 1=PCM or 3=IEEE float)",
              out->info.format_tag);
        return W2A_ERR_FORMAT;
    }
    if (out->info.channels < 1 || out->info.channels > 8) {
        LOG_E("unsupported channels=%u", out->info.channels);
        return W2A_ERR_FORMAT;
    }
    if (out->info.sample_rate < 8000 || out->info.sample_rate > 192000) {
        LOG_E("unsupported sample_rate=%u", out->info.sample_rate);
        return W2A_ERR_FORMAT;
    }
    if (out->info.format_tag == 1 &&
        out->info.bits_per_sample != 16 &&
        out->info.bits_per_sample != 24 &&
        out->info.bits_per_sample != 32) {
        LOG_E("unsupported PCM bits_per_sample=%u", out->info.bits_per_sample);
        return W2A_ERR_FORMAT;
    }
    if (out->info.format_tag == 3 && out->info.bits_per_sample != 32) {
        LOG_E("unsupported float bits_per_sample=%u (need 32)", out->info.bits_per_sample);
        return W2A_ERR_FORMAT;
    }

    /* Seek to start of data for caller convenience */
    if (fseek(fp, (long)out->info.data_offset, SEEK_SET) != 0) {
        return W2A_ERR_FORMAT;
    }

    LOG_I("WAV: %u Hz, %u ch, %u-bit, %s, %llu frames (%.2f s)",
          out->info.sample_rate,
          out->info.channels,
          out->info.bits_per_sample,
          out->info.format_tag == 1 ? "PCM" : "FLOAT",
          (unsigned long long)out->info.total_frames,
          (double)out->info.total_frames / (double)out->info.sample_rate);
    return W2A_OK;
}

void wav_close(wav_t* w) {
    if (!w || !w->fp) return;
    fclose(w->fp);
    w->fp = NULL;
}

int64_t wav_read_pcm(wav_t* w, void* dst, uint64_t frames) {
    if (!w || !w->fp || !dst) return W2A_ERR_ARGS;
    if (frames == 0) return 0;
    uint32_t ds = w->downsample ? w->downsample : 1;
    if (ds == 1) {
        size_t bytes = frames * (size_t)w->info.block_align;
        size_t got = fread(dst, 1, bytes, w->fp);
        if (got == 0 && feof(w->fp)) return 0;
        if (got < bytes && ferror(w->fp)) return W2A_ERR_READ;
        size_t frames_read = got / w->info.block_align;
        return (int64_t)frames_read;
    }
    /* Downsample: read N source frames per output frame, keep the first. */
    uint8_t* dst_bytes = (uint8_t*)dst;
    size_t out_ba = w->info.block_align;
    uint8_t* tmp = (uint8_t*)malloc(out_ba * ds);
    if (!tmp) return W2A_ERR_MEMORY;
    uint64_t out_frames = 0;
    for (uint64_t i = 0; i < frames; i++) {
        size_t got = fread(tmp, 1, out_ba * ds, w->fp);
        if (got < out_ba) {
            if (got == 0 && feof(w->fp)) break;
            free(tmp);
            return W2A_ERR_READ;
        }
        memcpy(dst_bytes + out_frames * out_ba, tmp, out_ba);
        out_frames++;
    }
    free(tmp);
    return (int64_t)out_frames;
}

void wav_set_downsample(wav_t* w, uint32_t factor) {
    if (!w) return;
    w->downsample = factor ? factor : 1;
}
uint32_t wav_downsample(const wav_t* w) {
    return w ? w->downsample : 1;
}

int wav_seek_frame(wav_t* w, uint64_t frame) {
    if (!w || !w->fp) return W2A_ERR_ARGS;
    long off = (long)(w->info.data_offset + (int64_t)frame * w->info.block_align);
    return fseek(w->fp, off, SEEK_SET) == 0 ? W2A_OK : W2A_ERR_READ;
}