#ifndef WAV2AAC_CONVERTER_H
#define WAV2AAC_CONVERTER_H

#include <stdint.h>
#include <stddef.h>

/*
 * Convert one WAV file to M4A (AAC-LC) with given bitrate.
 * Returns W2A_OK on success.
 */
int convert_wav_to_m4a(const char* in_path, const char* out_path, uint32_t bitrate);

/*
 * Batch convert many files using GCD dispatch_apply.
 * `paths` is array of input WAV paths; `out_paths` parallel output M4A paths.
 * `count` is number of files. `bitrate` in bps.
 * Returns W2A_OK if all succeed, otherwise returns the first error code.
 */
int convert_wav_batch(const char** paths, const char** out_paths,
                      size_t count, uint32_t bitrate);

#endif /* WAV2AAC_CONVERTER_H */