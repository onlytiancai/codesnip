/*
 * m4a_muxer.c — Minimal hand-rolled ISO BMFF (MPEG-4 Part 12) writer for
 * AAC-LC audio inside an M4A container. Supports the box set ftyp/moov/mdat
 * only; no edit lists, no chapters, no timecode tracks. Two-pass write:
 *
 *   Pass 1: caller appends raw AAC frames in memory
 *   Pass 2: finalize() writes ftyp + moov + mdat in one shot so stco offsets
 *           are correct without back-patching.
 *
 * All multi-byte integers are written big-endian.
 */

#include "m4a_muxer.h"
#include "util.h"
#include <arpa/inet.h>   /* htonl, htons */
#include <errno.h>
#include <stdlib.h>
#include <string.h>

/* ---------- Big-endian writers (local helpers) ---------- */

static int w_u8(FILE* fp, uint8_t v)  { return fputc(v, fp) != EOF ? 0 : -1; }
static int w_u16(FILE* fp, uint16_t v){ uint16_t be = htons(v);  return fwrite(&be, 2, 1, fp) == 1 ? 0 : -1; }
static int w_u32(FILE* fp, uint32_t v){ uint32_t be = htonl(v);  return fwrite(&be, 4, 1, fp) == 1 ? 0 : -1; }
static int w_bytes(FILE* fp, const void* p, size_t n){ return fwrite(p, 1, n, fp) == n ? 0 : -1; }
static int w_tag(FILE* fp, const char* t4){ return fwrite(t4, 1, 4, fp) == 4 ? 0 : -1; }

/* Box header = 4-byte size + 4-byte type. Returns current file offset of size field. */
static long w_box_open(FILE* fp, const char* type) {
    long off = ftell(fp);
    w_u32(fp, 0);          /* placeholder size */
    w_tag(fp, type);
    return off;
}
static void w_box_close(FILE* fp, long off) {
    long end = ftell(fp);
    uint32_t size = (uint32_t)(end - off);
    fseek(fp, off, SEEK_SET);
    w_u32(fp, size);
    fseek(fp, end, SEEK_SET);
}

/* ---------- AudioSpecificConfig (2 bytes for AAC-LC, SFI<15) ---------- */
static int write_asc(FILE* fp, uint32_t sample_rate, uint32_t channels) {
    static const uint32_t sfi_table[] = {
        96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050,
        16000, 12000, 11025, 8000,  7350,  0,     0,     0
    };
    uint32_t sfi = 0;
    for (uint32_t i = 0; i < 16; i++) {
        if (sfi_table[i] == sample_rate) { sfi = i; break; }
    }
    if (channels < 1 || channels > 7) {
        LOG_E("ASC: channel count %u out of range (1..7)", channels);
        return -1;
    }
    /* AOT_LC=2, GASpecificConfig: frameLengthFlag=0, dependsOnCoreCoder=0, extFlag=0 */
    uint16_t asc = (uint16_t)(((2u & 0x1f) << 11) | ((sfi & 0x0f) << 7) | ((channels & 0x0f) << 3));
    w_u16(fp, asc);
    return 0;
}

/* ---------- esds box contents (Elementary Stream Descriptor) ----------
 * ESDS extends FullBox, so version (1 byte) + flags (3 bytes) MUST precede
 * the ES_Descriptor. Apple's strict QuickTime parser rejects files that
 * skip this 4-byte header (treats the ES_Descriptor tag 0x03 as the esds
 * version byte, which is invalid). */
static long w_esds(FILE* fp, uint32_t sample_rate, uint32_t channels) {
    long esds_off = w_box_open(fp, "esds");
    w_u32(fp, 0);                      /* version=0, flags=0 (FullBox header) */
    /* ES_Descriptor (tag 0x03) */
    w_u8(fp, 0x03);
    {
        long len_off = ftell(fp);
        w_u8(fp, 0);
        long len_start = ftell(fp);
        w_u16(fp, 0);                    /* ES_ID */
        w_u8(fp, 0);                     /* flags/priority */
        /* DecoderConfigDescriptor (tag 0x04) */
        w_u8(fp, 0x04);
        {
            long dlen_off = ftell(fp);
            w_u8(fp, 0);
            long dlen_start = ftell(fp);
            w_u8(fp, 0x40);              /* objectTypeIndication = MPEG4_Audio */
            w_u8(fp, 0x15);              /* streamType=audio (5) | reserved(0)<<2 | upStream(0)<<1 */
            w_u8(fp, 0x00); w_u8(fp, 0x00); w_u8(fp, 0x00);  /* bufferSizeDB */
            w_u32(fp, 0);                /* maxBitrate */
            w_u32(fp, 0);                /* avgBitrate */
            /* DecoderSpecificInfo (tag 0x05) — wraps ASC */
            w_u8(fp, 0x05);
            {
                long ilen_off = ftell(fp);
                w_u8(fp, 0);
                long ilen_start = ftell(fp);
                if (write_asc(fp, sample_rate, channels) < 0) {
                    fseek(fp, esds_off, SEEK_SET);
                    w_box_close(fp, esds_off);
                    return -1;
                }
                long ilen_end = ftell(fp);
                fseek(fp, ilen_off, SEEK_SET);
                w_u8(fp, (uint8_t)(ilen_end - ilen_start));
                fseek(fp, ilen_end, SEEK_SET);
            }
            /* SLConfigDescriptor (tag 0x06) */
            w_u8(fp, 0x06);
            w_u8(fp, 1);
            w_u8(fp, 0x02);
            long dlen_end = ftell(fp);
            fseek(fp, dlen_off, SEEK_SET);
            w_u8(fp, (uint8_t)(dlen_end - dlen_start));
            fseek(fp, dlen_end, SEEK_SET);
        }
        long len_end = ftell(fp);
        fseek(fp, len_off, SEEK_SET);
        w_u8(fp, (uint8_t)(len_end - len_start));
        fseek(fp, len_end, SEEK_SET);
    }
    w_box_close(fp, esds_off);
    return esds_off;
}

/* ---------- ftyp ---------- */
static void w_ftyp(FILE* fp) {
    long off = w_box_open(fp, "ftyp");
    w_tag(fp, "M4A ");            /* major brand */
    w_u32(fp, 0);                 /* minor version */
    w_tag(fp, "M4A ");            /* compatible brand */
    w_tag(fp, "mp42");
    w_tag(fp, "isom");
    w_box_close(fp, off);
}

/* ---------- mvhd / tkhd / mdhd / hdlr / smhd / dinf ---------- */
static void w_mvhd(FILE* fp, uint64_t duration_ms) {
    long off = w_box_open(fp, "mvhd");
    w_u8(fp, 0);
    w_u8(fp, 0); w_u8(fp, 0); w_u8(fp, 0);  /* flags */
    w_u32(fp, 0); w_u32(fp, 0);            /* creation/mod time */
    w_u32(fp, 1000);                       /* timescale */
    w_u32(fp, (uint32_t)duration_ms);
    w_u32(fp, 0x00010000);                 /* rate = 1.0 */
    w_u16(fp, 0x0100);                     /* volume = 1.0 */
    w_u16(fp, 0);
    w_u32(fp, 0); w_u32(fp, 0);
    for (int i = 0; i < 9; i++) w_u32(fp, 0);  /* unity matrix */
    w_u32(fp, 0); w_u32(fp, 0); w_u32(fp, 0); w_u32(fp, 0);
    w_u32(fp, 0); w_u32(fp, 0);
    w_u32(fp, 2);                          /* pre_defined */
    w_u32(fp, 0);                          /* next_track_ID */
    w_box_close(fp, off);
}

static void w_tkhd(FILE* fp, uint64_t duration, uint32_t track_id) {
    long off = w_box_open(fp, "tkhd");
    w_u8(fp, 0);
    w_u8(fp, 0); w_u8(fp, 0); w_u8(fp, 0x07);  /* enabled+in_movie+in_preview */
    w_u32(fp, 0); w_u32(fp, 0);                /* create/mod time */
    w_u32(fp, track_id);
    w_u32(fp, 0);                              /* reserved */
    w_u32(fp, (uint32_t)duration);
    w_u32(fp, 0); w_u32(fp, 0);
    w_u16(fp, 0);                              /* layer */
    w_u16(fp, 0);                              /* alternate_group */
    w_u16(fp, 0x0100);                         /* volume */
    w_u16(fp, 0);
    for (int i = 0; i < 9; i++) w_u32(fp, 0);  /* matrix */
    w_u32(fp, 0); w_u32(fp, 0); w_u32(fp, 0); w_u32(fp, 0);
    w_u32(fp, 0); w_u32(fp, 0);
    w_u32(fp, 0);                              /* width */
    w_u32(fp, 0);                              /* height */
    w_box_close(fp, off);
}

static void w_mdhd(FILE* fp, uint32_t sample_rate, uint64_t total_frames) {
    long off = w_box_open(fp, "mdhd");
    w_u8(fp, 0);
    w_u8(fp, 0); w_u8(fp, 0); w_u8(fp, 0);
    w_u32(fp, 0); w_u32(fp, 0);
    w_u32(fp, sample_rate);
    w_u32(fp, (uint32_t)total_frames);
    w_u16(fp, 0x55c4);                  /* language = 'und' */
    w_u16(fp, 0);
    w_box_close(fp, off);
}

static void w_hdlr(FILE* fp) {
    long off = w_box_open(fp, "hdlr");
    w_u8(fp, 0);
    w_u8(fp, 0); w_u8(fp, 0); w_u8(fp, 0);
    w_u32(fp, 0);
    w_tag(fp, "soun");
    w_u32(fp, 0); w_u32(fp, 0); w_u32(fp, 0);
    w_u8(fp, 0);                        /* name = empty C-string */
    w_box_close(fp, off);
}

static void w_smhd(FILE* fp) {
    long off = w_box_open(fp, "smhd");
    w_u8(fp, 0);
    w_u8(fp, 0); w_u8(fp, 0); w_u8(fp, 0);
    w_u16(fp, 0);
    w_u16(fp, 0);
    w_box_close(fp, off);
}

static void w_dinf(FILE* fp) {
    long off = w_box_open(fp, "dinf");
    long dref_off = w_box_open(fp, "dref");
    w_u32(fp, 0);
    w_u32(fp, 1);
    {
        long url_off = w_box_open(fp, "url ");
        w_u8(fp, 0); w_u8(fp, 0); w_u8(fp, 0); w_u8(fp, 1);  /* self-contained */
        w_box_close(fp, url_off);
    }
    w_box_close(fp, dref_off);
    w_box_close(fp, off);
}

/* ---------- Sample entry: mp4a (audio) + esds ---------- */
static void w_audio_sample_entry(FILE* fp, uint32_t sample_rate, uint32_t channels,
                                 uint32_t bitrate) {
    (void)bitrate;
    long off = w_box_open(fp, "mp4a");
    w_u32(fp, 0); w_u16(fp, 0);              /* reserved */
    w_u16(fp, 1);                            /* data_reference_index */
    w_u32(fp, 0); w_u32(fp, 0);              /* reserved x2 */
    w_u16(fp, (uint16_t)channels);           /* channelcount */
    w_u16(fp, 16);                           /* samplesize */
    w_u16(fp, 0);                            /* pre_defined */
    w_u16(fp, 0);                            /* reserved */
    w_u32(fp, sample_rate << 16);            /* samplerate 16.16 */
    w_esds(fp, sample_rate, channels);
    w_box_close(fp, off);
}

/* ---------- Sample table ---------- */
static void w_stsd(FILE* fp, m4a_mux_t* m) {
    long off = w_box_open(fp, "stsd");
    w_u8(fp, 0);
    w_u8(fp, 0); w_u8(fp, 0); w_u8(fp, 0);
    w_u32(fp, 1);
    w_audio_sample_entry(fp, m->sample_rate, m->channels, m->bitrate);
    w_box_close(fp, off);
}

static void w_stts(FILE* fp, uint64_t frame_count) {
    long off = w_box_open(fp, "stts");
    w_u8(fp, 0);
    w_u8(fp, 0); w_u8(fp, 0); w_u8(fp, 0);
    w_u32(fp, 1);
    w_u32(fp, (uint32_t)frame_count);
    w_u32(fp, 1024);
    w_box_close(fp, off);
}

static void w_stsc(FILE* fp, uint64_t frame_count) {
    long off = w_box_open(fp, "stsc");
    w_u8(fp, 0);
    w_u8(fp, 0); w_u8(fp, 0); w_u8(fp, 0);
    w_u32(fp, 1);
    w_u32(fp, 1);
    w_u32(fp, (uint32_t)frame_count);
    w_u32(fp, 1);
    w_box_close(fp, off);
}

static void w_stsz(FILE* fp, const m4a_mux_t* m) {
    long off = w_box_open(fp, "stsz");
    w_u8(fp, 0);
    w_u8(fp, 0); w_u8(fp, 0); w_u8(fp, 0);
    w_u32(fp, 0);                                /* variable sample size */
    w_u32(fp, (uint32_t)m->frame_count);
    for (uint64_t i = 0; i < m->frame_count; i++) {
        w_u32(fp, m->frame_sizes[i]);
    }
    w_box_close(fp, off);
}

/* stco writer that returns the file offset of the first chunk-offset entry
 * so the caller can patch it once the mdat payload offset is known. */
static long w_stco(FILE* fp, uint64_t mdat_data_offset, const m4a_mux_t* m) {
    long off = w_box_open(fp, "stco");
    w_u8(fp, 0);
    w_u8(fp, 0); w_u8(fp, 0); w_u8(fp, 0);
    w_u32(fp, (uint32_t)m->frame_count);
    long first_entry_off = ftell(fp);
    uint64_t cur = mdat_data_offset;
    for (uint64_t i = 0; i < m->frame_count; i++) {
        w_u32(fp, (uint32_t)cur);
        cur += m->frame_sizes[i];
    }
    w_box_close(fp, off);
    return first_entry_off;
}

static void w_stbl(FILE* fp, m4a_mux_t* m, long* stco_entries_off) {
    long off = w_box_open(fp, "stbl");
    w_stsd(fp, m);
    w_stts(fp, m->frame_count);
    w_stsc(fp, m->frame_count);
    w_stsz(fp, m);
    /* Pass 0 as mdat_data_offset; the caller will patch stco entries later. */
    *stco_entries_off = w_stco(fp, 0, m);
    w_box_close(fp, off);
}

static void w_minf(FILE* fp, m4a_mux_t* m, long* stco_entries_off) {
    long off = w_box_open(fp, "minf");
    w_smhd(fp);
    w_dinf(fp);
    w_stbl(fp, m, stco_entries_off);
    w_box_close(fp, off);
}

static void w_mdia(FILE* fp, m4a_mux_t* m, long* stco_entries_off) {
    long off = w_box_open(fp, "mdia");
    w_mdhd(fp, m->sample_rate, m->total_pcm_frames);
    w_hdlr(fp);
    w_minf(fp, m, stco_entries_off);
    w_box_close(fp, off);
}

static void w_trak(FILE* fp, m4a_mux_t* m, long* stco_entries_off) {
    long off = w_box_open(fp, "trak");
    /* tkhd duration is in *mvhd* timescale (1000 = ms), not mdhd timescale.
     * Previously passed m->total_pcm_frames (PCM frames at sample_rate),
     * which produced a nonsensical 5.78-day duration. */
    uint64_t dur_ms = (m->sample_rate > 0)
        ? (m->total_pcm_frames * 1000ULL) / m->sample_rate
        : 0;
    w_tkhd(fp, dur_ms, 1);
    w_mdia(fp, m, stco_entries_off);
    w_box_close(fp, off);
}

static void w_moov(FILE* fp, m4a_mux_t* m, long* stco_entries_off) {
    long off = w_box_open(fp, "moov");
    uint64_t dur_ms = (m->sample_rate > 0)
        ? (m->total_pcm_frames * 1000ULL) / m->sample_rate
        : 0;
    w_mvhd(fp, dur_ms);
    w_trak(fp, m, stco_entries_off);
    w_box_close(fp, off);
}

/* ---------- Public API ---------- */

int m4a_mux_open(m4a_mux_t* m, const char* path,
                 uint32_t sample_rate, uint32_t channels, uint32_t bitrate) {
    if (!m || !path) return W2A_ERR_ARGS;
    memset(m, 0, sizeof(*m));
    m->sample_rate = sample_rate;
    m->channels    = channels;
    m->bitrate     = bitrate;
    m->frame_cap   = 1024;
    m->frames      = (uint8_t**)calloc(m->frame_cap, sizeof(uint8_t*));
    m->frame_sizes = (uint32_t*)calloc(m->frame_cap, sizeof(uint32_t));
    if (!m->frames || !m->frame_sizes) {
        free(m->frames); free(m->frame_sizes);
        return W2A_ERR_MEMORY;
    }
    m->fp = fopen(path, "wb");
    if (!m->fp) {
        LOG_E("fopen(%s) for write failed: %s", path, strerror(errno));
        return W2A_ERR_OPEN;
    }
    return W2A_OK;
}

int m4a_mux_append_frame(m4a_mux_t* m, const uint8_t* data, uint32_t size) {
    if (!m || !data || size == 0) return W2A_ERR_ARGS;
    if (m->frame_count == m->frame_cap) {
        uint64_t new_cap = m->frame_cap * 2;
        uint8_t**  nf = (uint8_t**)realloc(m->frames, new_cap * sizeof(uint8_t*));
        uint32_t* ns = (uint32_t*)realloc(m->frame_sizes, new_cap * sizeof(uint32_t));
        if (!nf || !ns) {
            if (nf) m->frames = nf;
            return W2A_ERR_MEMORY;
        }
        m->frames = nf;
        m->frame_sizes = ns;
        m->frame_cap = new_cap;
    }
    uint8_t* copy = (uint8_t*)malloc(size);
    if (!copy) return W2A_ERR_MEMORY;
    memcpy(copy, data, size);
    m->frames[m->frame_count]      = copy;
    m->frame_sizes[m->frame_count]  = size;
    m->frame_count++;
    return W2A_OK;
}

int m4a_mux_finalize(m4a_mux_t* m) {
    if (!m || !m->fp) return W2A_ERR_ARGS;
    if (m->frame_count == 0) {
        LOG_W("finalize called with zero AAC frames");
    }
    /* Pass 1: write ftyp */
    w_ftyp(m->fp);
    /* Pass 2: write moov (with placeholder stco offsets) */
    long stco_entries_off = 0;
    w_moov(m->fp, m, &stco_entries_off);
    /* Now we know the mdat payload offset = current ftell + 8 (mdat box header) */
    long mdat_header_off = ftell(m->fp);
    uint64_t mdat_payload_off = (uint64_t)mdat_header_off + 8ULL;
    /* Patch stco entries in-place */
    if (m->frame_count > 0) {
        fseek(m->fp, stco_entries_off, SEEK_SET);
        uint64_t cur = mdat_payload_off;
        for (uint64_t i = 0; i < m->frame_count; i++) {
            w_u32(m->fp, (uint32_t)cur);
            cur += m->frame_sizes[i];
        }
    }
    /* Pass 3: write mdat box header + payload */
    fseek(m->fp, mdat_header_off, SEEK_SET);
    w_u32(m->fp, 0);                          /* placeholder mdat size */
    w_tag(m->fp, "mdat");
    for (uint64_t i = 0; i < m->frame_count; i++) {
        w_bytes(m->fp, m->frames[i], m->frame_sizes[i]);
    }
    long mdat_end = ftell(m->fp);
    uint32_t mdat_size = (uint32_t)(mdat_end - mdat_header_off);
    fseek(m->fp, mdat_header_off, SEEK_SET);
    w_u32(m->fp, mdat_size);
    fflush(m->fp);
    LOG_I("M4A: wrote %llu AAC frames, %llu PCM frames, %.2f s",
          (unsigned long long)m->frame_count,
          (unsigned long long)m->total_pcm_frames,
          m->sample_rate ? (double)m->total_pcm_frames / m->sample_rate : 0.0);
    return W2A_OK;
}

void m4a_mux_close(m4a_mux_t* m) {
    if (!m) return;
    if (m->fp) { fclose(m->fp); m->fp = NULL; }
    if (m->frames) {
        for (uint64_t i = 0; i < m->frame_count; i++) free(m->frames[i]);
        free(m->frames);
    }
    free(m->frame_sizes);
    memset(m, 0, sizeof(*m));
}