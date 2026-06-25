#include "util.h"
#include <time.h>
#include <stdarg.h>

static log_level_t g_log_level = LOG_INFO;

void log_set_level(log_level_t level) { g_log_level = level; }

void log_msg(log_level_t level, const char* fmt, ...) {
    if (level < g_log_level) return;
    static const char* tag[] = {"DEBUG", "INFO ", "WARN ", "ERROR"};
    fprintf(stderr, "[%s] ", tag[level]);
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
}

const char* w2a_strerror(int err) {
    switch (err) {
        case W2A_OK:           return "OK";
        case W2A_ERR_OPEN:     return "file open failed";
        case W2A_ERR_READ:     return "file read failed";
        case W2A_ERR_FORMAT:   return "invalid audio format";
        case W2A_ERR_ENCODE:   return "AAC encode failed";
        case W2A_ERR_MUX:      return "M4A mux failed";
        case W2A_ERR_ARGS:     return "invalid arguments";
        case W2A_ERR_MEMORY:   return "memory allocation failed";
        case W2A_ERR_THREAD:   return "threading error";
        default:               return "unknown error";
    }
}