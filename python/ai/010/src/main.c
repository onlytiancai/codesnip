/*
 * main.c — CLI entry point.
 *
 * Usage:
 *   wav2aac <in.wav> <out.m4a>                  # single file
 *   wav2aac --batch <list.txt> <out_dir>        # batch from newline list
 *   wav2aac --batch-glob <pattern> <out_dir>    # batch via glob
 *   wav2aac --info                              # show encoder info
 *   wav2aac -b <bitrate_kbps> ...               # override bitrate (default 128)
 */

#include "util.h"
#include "converter.h"
#include "aac_encoder.h"
#include "wav_parser.h"

#include <dirent.h>
#include <fnmatch.h>
#include <libgen.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <glob.h>

static void print_usage(const char* prog) {
    fprintf(stderr,
        "Usage:\n"
        "  %s <in.wav> <out.m4a> [bitrate_kbps]\n"
        "  %s --batch <list.txt> <out_dir> [bitrate_kbps]\n"
        "  %s --batch-glob <pattern> <out_dir> [bitrate_kbps]\n"
        "  %s --info\n"
        "\nDefaults: bitrate=128 kbps, AAC-LC, .m4a container.\n",
        prog, prog, prog, prog);
}

static int run_info(void) {
    /* Create a transient encoder at 48k stereo 128k to print available components. */
    aac_enc_t enc;
    int rc = aac_enc_create(&enc, 48000.0, 2, 128000);
    if (rc != W2A_OK) return rc;
    aac_enc_print_info(&enc);
    aac_enc_destroy(&enc);
    return W2A_OK;
}

static int ensure_dir(const char* path) {
    struct stat st;
    if (stat(path, &st) == 0) return S_ISDIR(st.st_mode) ? 0 : -1;
    return mkdir(path, 0755);
}

static int ends_with(const char* s, const char* suf) {
    size_t ls = strlen(s), lf = strlen(suf);
    return ls >= lf && strcasecmp(s + ls - lf, suf) == 0;
}

static char* derive_out_path(const char* in_path, const char* out_dir) {
    const char* base = strrchr(in_path, '/');
    base = base ? base + 1 : in_path;
    size_t blen = strlen(base);
    char* name = strdup(base);
    if (!name) return NULL;
    if (ends_with(name, ".wav")) name[blen - 4] = '\0';
    size_t need = strlen(out_dir) + 1 + strlen(name) + 4 + 1; /* / + .m4a + NUL */
    char* out = (char*)malloc(need);
    if (!out) { free(name); return NULL; }
    snprintf(out, need, "%s/%s.m4a", out_dir, name);
    free(name);
    return out;
}

int main(int argc, char** argv) {
    if (argc < 2) { print_usage(argv[0]); return 1; }

    /* Handle --info */
    if (argc == 2 && strcmp(argv[1], "--info") == 0) {
        return run_info() == W2A_OK ? 0 : 1;
    }

    /* Default log level: WARN. Use --debug to enable INFO. */
    int arg_shift = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--debug") == 0) { log_set_level(LOG_INFO); arg_shift++; continue; }
        if (strcmp(argv[i], "-v") == 0)       { log_set_level(LOG_DEBUG); arg_shift++; continue; }
    }
    /* Shift argv past the flags we consumed (we only need to inspect up to 1 flag) */
    if (arg_shift > 0) {
        /* Build a fresh argv */
        char** nv = (char**)calloc((size_t)(argc - arg_shift), sizeof(char*));
        nv[0] = argv[0];
        int j = 1;
        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--debug") == 0 || strcmp(argv[i], "-v") == 0) continue;
            nv[j++] = argv[i];
        }
        argc -= arg_shift;
        argv = nv;
    }

    if (argc >= 3 && strcmp(argv[1], "--batch") == 0) {
        /* Read newline list of WAVs */
        const char* list_path = argv[2];
        const char* out_dir   = argv[3];
        uint32_t br = (argc >= 5) ? (uint32_t)atoi(argv[4]) * 1000U : 128000U;
        if (ensure_dir(out_dir) < 0) {
            fprintf(stderr, "out_dir %s is not a directory\n", out_dir);
            return 1;
        }
        FILE* f = fopen(list_path, "r");
        if (!f) { perror("open list"); return 1; }
        char line[1024];
        char* paths[4096];
        char* outs[4096];
        size_t n = 0;
        while (fgets(line, sizeof(line), f) && n < 4096) {
            size_t L = strlen(line);
            while (L > 0 && (line[L-1] == '\n' || line[L-1] == '\r')) line[--L] = 0;
            if (L == 0) continue;
            paths[n] = strdup(line);
            outs[n]  = derive_out_path(line, out_dir);
            if (!paths[n] || !outs[n]) { fclose(f); return 1; }
            n++;
        }
        fclose(f);
        LOG_I("batch: %zu files, %u bps, out=%s", n, br, out_dir);
        int rc = convert_wav_batch((const char**)paths, (const char**)outs, n, br);
        for (size_t i = 0; i < n; i++) { free(paths[i]); free(outs[i]); }
        return rc == W2A_OK ? 0 : 1;
    }

    if (argc >= 4 && strcmp(argv[1], "--batch-glob") == 0) {
        const char* pattern = argv[2];
        const char* out_dir = argv[3];
        uint32_t br = (argc >= 5) ? (uint32_t)atoi(argv[4]) * 1000U : 128000U;
        if (ensure_dir(out_dir) < 0) { fprintf(stderr, "out_dir bad\n"); return 1; }
        glob_t g;
        if (glob(pattern, 0, NULL, &g) != 0 || g.gl_pathc == 0) {
            fprintf(stderr, "glob %s matched nothing\n", pattern);
            return 1;
        }
        const char* paths[4096];
        const char* outs[4096];
        size_t n = g.gl_pathc < 4096 ? g.gl_pathc : 4096;
        for (size_t i = 0; i < n; i++) {
            paths[i] = g.gl_pathv[i];
            outs[i]  = derive_out_path(g.gl_pathv[i], out_dir);
        }
        LOG_I("glob batch: %zu files", n);
        int rc = convert_wav_batch(paths, outs, n, br);
        for (size_t i = 0; i < n; i++) free((void*)outs[i]);
        globfree(&g);
        return rc == W2A_OK ? 0 : 1;
    }

    /* Single file */
    if (argc < 3) { print_usage(argv[0]); return 1; }
    const char* in_path  = argv[1];
    const char* out_path = argv[2];
    uint32_t br = (argc >= 4) ? (uint32_t)atoi(argv[3]) * 1000U : 128000U;

    double t0 = 0.0, t1 = 0.0;
    extern double bench_now(void);
    t0 = bench_now();
    int rc = convert_wav_to_m4a(in_path, out_path, br);
    t1 = bench_now();
    if (rc == W2A_OK) {
        LOG_I("OK: %s -> %s in %.3f s", in_path, out_path, t1 - t0);
    } else {
        LOG_E("FAILED: %s -> %s: %s", in_path, out_path, w2a_strerror(rc));
    }
    return rc == W2A_OK ? 0 : 1;
}