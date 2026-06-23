#ifndef WAV2AAC_UTIL_H
#define WAV2AAC_UTIL_H

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Log levels */
typedef enum {
    LOG_DEBUG = 0,
    LOG_INFO  = 1,
    LOG_WARN  = 2,
    LOG_ERROR = 3,
} log_level_t;

void log_set_level(log_level_t level);
void log_msg(log_level_t level, const char* fmt, ...);

/* Shortcut macros */
#define LOG_D(...) log_msg(LOG_DEBUG, __VA_ARGS__)
#define LOG_I(...) log_msg(LOG_INFO,  __VA_ARGS__)
#define LOG_W(...) log_msg(LOG_WARN,  __VA_ARGS__)
#define LOG_E(...) log_msg(LOG_ERROR, __VA_ARGS__)

/* Error codes */
#define W2A_OK             0
#define W2A_ERR_OPEN     (-1)
#define W2A_ERR_READ     (-2)
#define W2A_ERR_FORMAT   (-3)
#define W2A_ERR_ENCODE   (-4)
#define W2A_ERR_MUX      (-5)
#define W2A_ERR_ARGS     (-6)
#define W2A_ERR_MEMORY   (-7)
#define W2A_ERR_THREAD   (-8)

const char* w2a_strerror(int err);

/* End-Of-Media-Gap custom error code (FourCC) for converter EOS signalling */
#define EOMG_FOURCC ('E'<<24 | 'O'<<16 | 'M'<<8 | 'G')

#endif /* WAV2AAC_UTIL_H */