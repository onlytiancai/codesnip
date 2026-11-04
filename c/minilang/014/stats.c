/*
 * stats.c —— 计算一组数值的最大值、最小值、平均值、中位数和任意百分位数
 *
 * 编译:
 *   cc -std=c11 -O2 -Wall -Wextra -pedantic -o stats stats.c -lm
 *
 * 用法:
 *   ./stats 1 2 3 4 5                        # 命令行直接给数
 *   seq 1 100 | ./stats                     # 从标准输入读（空白/逗号/分号分隔，支持多行）
 *   echo -n '100\n50\n22\n' | ./stats       # bash 未解释的字面 \n 也能识别
 *   ./stats -f data.txt                     # 从文件读，'#' 之后为注释
 *   ./stats -f data.txt -m nearest          # 用最近秩算法（监控系统常用）
 *   ./stats -p 99.9,99.5,75,25 -f data.txt  # 自定义百分位列表
 *   seq 1 100 | ./stats -c                  # 输出两行 CSV，便于脚本处理
 *   seq 1 2000000 | ./stats -n 0 -c         # 取消数量上限
 *   ./stats -h                              # 查看帮助
 *
 * 默认最多接收 100 万个数值，超出即报错退出；-n / --max 可调整，0 表示不限制。
 *
 * 百分位算法（不同工具结果不一致，主要坑就在这里）:
 *   linear  (默认): h = (n-1)*p/100，在相邻两个样本间线性插值。
 *                   等价于 numpy.percentile 默认 / Excel PERCENTILE.INC / R type 7。
 *   nearest       : 取排序后第 ceil(p/100*n) 个数（1 起算），不做插值。
 *                   等价于最近秩法，Prometheus / 多数监控面板的口径。
 *   median 固定为标准定义：n 为偶数时取中间两数的算术平均（与 linear 的 p50 相同）。
 *
 * 输入数值必须是有限十进制数（支持科学计数法、负数），NaN/Inf 与非法 token 会报错。
 */

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#  include <io.h>
#  define IS_TTY(f) _isatty(_fileno(f))
#else
#  include <unistd.h>
#  define IS_TTY(f) isatty(fileno(f))
#endif

#define DEFAULT_PERCENTILES "85,90,95,99"
#define DEFAULT_MAX_VALUES 1000000u   /* 默认数值个数上限；0 表示不限制 */
#define MAX_PERCENTILES 64
#define NUM_FMT "%.10g"          /* 统一数值输出格式 */
#define DELIMS " \t\r\n,;"

typedef struct {
    double *data;
    size_t  len;
    size_t  cap;
    size_t  max;               /* 允许的最大数值个数，0 = 不限制 */
} Vec;

static void die(const char *msg)
{
    fprintf(stderr, "错误: %s\n", msg);
    exit(1);
}

/* 返回 0 成功，-1 内存不足，-2 超出上限 */
static int vec_push(Vec *v, double x)
{
    if (v->max != 0 && v->len >= v->max) return -2;
    if (v->len == v->cap) {
        size_t cap = v->cap ? v->cap * 2 : 128;
        double *p = realloc(v->data, cap * sizeof *p);
        if (!p) return -1;
        v->data = p;
        v->cap = cap;
    }
    v->data[v->len++] = x;
    return 0;
}

/* 统一处理 push 错误，让 CLI 与文件/stdin 读取给出一致提示 */
static void push_or_die(Vec *v, double x)
{
    int r = vec_push(v, x);
    if (r == -1) die("内存不足");
    if (r == -2) {
        fprintf(stderr,
                "错误: 数值个数超过上限 %zu（用 -n/--max 调整，0 表示不限制）\n",
                v->max);
        exit(1);
    }
}

/* 解析 -n 上限：非负整数，0 表示不限制 */
static size_t parse_limit(const char *s)
{
    char *end;
    unsigned long long val;

    errno = 0;
    if (*s == '-') die("-n 需要非负整数（0 表示不限制）");
    val = strtoull(s, &end, 10);
    if (end == s || *end != '\0' || errno == ERANGE || val > (unsigned long long)SIZE_MAX)
        die("-n 需要非负整数（0 表示不限制）");
    return (size_t)val;
}

/* 严格解析：整个 token 都必须是一个有限数字，不允许 "12abc"、nan、inf */
static int parse_number(const char *s, double *out)
{
    char *end;
    double x;

    errno = 0;
    x = strtod(s, &end);
    if (end == s || *end != '\0') return -1;
    if (errno == ERANGE || !isfinite(x)) return -2;
    *out = x;
    return 0;
}

/* 动态读取一行，不截断超长行；返回 NULL 表示 EOF */
static char *read_line(FILE *fp)
{
    size_t cap = 256, len = 0;
    char *buf = malloc(cap);
    int c;

    if (!buf) die("内存不足");
    while ((c = fgetc(fp)) != EOF) {
        if (len + 1 >= cap) {
            char *p;
            cap *= 2;
            p = realloc(buf, cap);
            if (!p) { free(buf); die("内存不足"); }
            buf = p;
        }
        buf[len++] = (char)c;
        if (c == '\n') break;
    }
    if (len == 0) { free(buf); return NULL; }
    buf[len] = '\0';
    return buf;
}

/* 兼容 shell 未解释的字面转义：bash 下 `echo -n '100\n50\n'` 输出的是
 * 字面 \n 而非换行，这里把它（以及 \t / \r）归一化成空白分隔符。
 * 合法数值不含反斜杠，因此替换是安全的；就地压缩不会增长缓冲区。 */
static void normalize_escapes(char *s)
{
    char *src = s, *dst = s;

    while (*src) {
        if (src[0] == '\\' && (src[1] == 'n' || src[1] == 't' || src[1] == 'r')) {
            *dst++ = ' ';
            src += 2;
        } else {
            *dst++ = *src++;
        }
    }
    *dst = '\0';
}

static void read_stream(Vec *v, FILE *fp, const char *name)
{
    char *line;

    while ((line = read_line(fp)) != NULL) {
        char *hash = strchr(line, '#');
        char *tok;
        if (hash) *hash = '\0';                 /* '#' 起为注释 */
        normalize_escapes(line);
        for (tok = strtok(line, DELIMS); tok; tok = strtok(NULL, DELIMS)) {
            double x;
            int r = parse_number(tok, &x);
            if (r != 0) {
                fprintf(stderr, "错误: %s 中数值 '%s' %s\n", name, tok,
                        r == -2 ? "溢出或不是有限数" : "无法解析");
                free(line);
                exit(1);
            }
            push_or_die(v, x);
        }
        free(line);
    }
    if (ferror(fp)) die("读取输入失败");
}

static int cmp_double(const void *pa, const void *pb)
{
    double a = *(const double *)pa, b = *(const double *)pb;
    return (a > b) - (a < b);
}

/* 线性插值法：h = (n-1)*p/100 */
static double pct_linear(const double *a, size_t n, double p)
{
    double h, frac;
    size_t lo;

    if (n == 1) return a[0];
    h = (double)(n - 1) * p / 100.0;
    lo = (size_t)h;
    frac = h - (double)lo;
    if (lo + 1 >= n) return a[n - 1];
    /* 异号且都接近 DBL_MAX 时，a[hi]-a[lo] 会溢出，改用凸组合 */
    if ((a[lo] < 0.0) != (a[lo + 1] < 0.0))
        return (1.0 - frac) * a[lo] + frac * a[lo + 1];
    return a[lo] + frac * (a[lo + 1] - a[lo]);
}

/* 最近秩法：第 ceil(p/100*n) 个样本；减去微小容差规避浮点误差（如 0.85*100） */
static double pct_nearest(const double *a, size_t n, double p)
{
    double r = p / 100.0 * (double)n;
    size_t idx = (size_t)ceil(r - 1e-9);
    if (idx < 1) idx = 1;
    if (idx > n) idx = n;
    return a[idx - 1];
}

/* 求平均值：先按最大绝对值缩放，再做 Kahan 补偿求和。
 * 缩放可避免大数累加溢出（部分平台 long double 与 double 同宽），
 * 补偿求和可避免大量数据累加时的精度损失。 */
static double mean_of(const double *a, size_t n)
{
    double scale = 0.0;
    long double sum = 0.0L, c = 0.0L;
    size_t i;

    for (i = 0; i < n; i++) {
        double t = fabs(a[i]);
        if (t > scale) scale = t;
    }
    if (scale == 0.0) return 0.0;

    for (i = 0; i < n; i++) {
        long double y = (long double)a[i] / (long double)scale - c;
        long double t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    return (double)((sum / (long double)n) * (long double)scale);
}

/* 标准中位数：偶数个样本取中间两数平均，先各自减半避免相加溢出 */
static double median_of(const double *a, size_t n)
{
    return (n % 2) ? a[n / 2] : a[n / 2 - 1] / 2.0 + a[n / 2] / 2.0;
}

static void pct_label(double p, char *buf, size_t sz)
{
    snprintf(buf, sz, "p%.10g", p);
}

/* 解析 -p 列表，去重并保持用户给定顺序 */
static size_t parse_pct_list(const char *s, double *out, size_t max)
{
    char *buf = malloc(strlen(s) + 1);
    char *tok;
    size_t n = 0;

    if (!buf) die("内存不足");
    strcpy(buf, s);
    for (tok = strtok(buf, ", \t"); tok; tok = strtok(NULL, ", \t")) {
        double p;
        size_t j;
        int dup = 0;

        if (parse_number(tok, &p) != 0 || p < 0.0 || p > 100.0) {
            fprintf(stderr, "错误: 百分位 '%s' 非法（应为 0~100 之间的数）\n", tok);
            free(buf);
            exit(1);
        }
        for (j = 0; j < n; j++) if (out[j] == p) { dup = 1; break; }
        if (!dup) {
            if (n >= max) { free(buf); die("百分位个数超过上限"); }
            out[n++] = p;
        }
    }
    free(buf);
    if (n == 0) die("百分位列表为空");
    return n;
}

static void usage(const char *prog)
{
    printf(
        "用法: %s [选项] [数值 ...]\n"
        "\n"
        "数据来源（可混合使用）:\n"
        "  数值 ...              命令行参数，支持负数与科学计数法\n"
        "  -f, --file PATH       从文件读取，'-' 表示标准输入；'#' 之后为注释\n"
        "  （无参数时）          从标准输入读取，支持多行；空白/逗号/分号分隔，\n"
        "                        字面的 \\n / \\t / \\r 也被当作空白\n"
        "\n"
        "选项:\n"
        "  -p, --percentiles LIST  逗号分隔的百分位，默认 " DEFAULT_PERCENTILES "\n"
        "  -m, --method METHOD     linear（默认，线性插值）或 nearest（最近秩）\n"
        "  -c, --csv              输出两行 CSV（表头 + 数值），便于脚本解析\n"
        "  -n, --max N            数值个数上限，默认 1000000；0 表示不限制\n"
        "  -h, --help             显示本帮助\n"
        "\n"
        "输出: count / min / max / mean / median / 各百分位 / 所用算法\n"
        "示例: seq 1 100 | %s -p 50,85,90,95,99,99.9\n",
        prog, prog);
}

int main(int argc, char **argv)
{
    const char *pct_spec = DEFAULT_PERCENTILES;
    const char *method = "linear";
    int nearest = 0, csv = 0, got_file = 0;
    double pcts[MAX_PERCENTILES];
    size_t npct, i, n;
    double *a, mean, median;
    Vec v = { NULL, 0, 0, DEFAULT_MAX_VALUES };

    for (i = 1; i < (size_t)argc; i++) {
        const char *arg = argv[i];
        double x;

        /* 先当数字试，负数（如 -5）因此不会被误认成选项 */
        if (parse_number(arg, &x) == 0) {
            push_or_die(&v, x);
            continue;
        }
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            usage(argv[0]);
            free(v.data);
            return 0;
        }
        if (!strcmp(arg, "-c") || !strcmp(arg, "--csv")) {
            csv = 1;
        } else if (!strcmp(arg, "-f") || !strcmp(arg, "--file")) {
            FILE *fp;
            if (++i >= (size_t)argc) die("-f 需要跟一个文件名");
            fp = !strcmp(argv[i], "-") ? stdin : fopen(argv[i], "r");
            if (!fp) {
                fprintf(stderr, "错误: 无法打开 '%s': %s\n", argv[i], strerror(errno));
                free(v.data);
                return 1;
            }
            read_stream(&v, fp, argv[i]);
            if (fp != stdin) fclose(fp);
            got_file = 1;
        } else if (!strcmp(arg, "-p") || !strcmp(arg, "--percentiles")) {
            if (++i >= (size_t)argc) die("-p 需要跟一个百分位列表");
            pct_spec = argv[i];
        } else if (!strcmp(arg, "-m") || !strcmp(arg, "--method")) {
            if (++i >= (size_t)argc) die("-m 需要跟 linear 或 nearest");
            if (!strcmp(argv[i], "nearest")) { nearest = 1; method = "nearest"; }
            else if (!strcmp(argv[i], "linear")) { nearest = 0; method = "linear"; }
            else die("-m 只接受 linear 或 nearest");
        } else if (!strcmp(arg, "-n") || !strcmp(arg, "--max")) {
            if (++i >= (size_t)argc) die("-n 需要跟一个非负整数");
            v.max = parse_limit(argv[i]);
            /* 允许 -n 出现在数值之后，此时要立即校验已有数据 */
            if (v.max != 0 && v.len >= v.max) {
                fprintf(stderr,
                        "错误: 数值个数超过上限 %zu（用 -n/--max 调整，0 表示不限制）\n",
                        v.max);
                free(v.data);
                return 1;
            }
        } else {
            fprintf(stderr, "错误: 未知选项或非法数值 '%s'\n", arg);
            usage(argv[0]);
            free(v.data);
            return 1;
        }
    }

    if (v.len == 0 && !got_file) {
        if (IS_TTY(stdin)) {
            usage(argv[0]);
            fprintf(stderr, "错误: 没有输入数据\n");
            free(v.data);
            return 1;
        }
        read_stream(&v, stdin, "<stdin>");
    }
    if (v.len == 0) die("输入中没有有效数值");

    npct = parse_pct_list(pct_spec, pcts, MAX_PERCENTILES);
    a = v.data;
    n = v.len;
    qsort(a, n, sizeof *a, cmp_double);

    mean = mean_of(a, n);
    median = median_of(a, n);

    if (csv) {
        printf("count,min,max,mean,median");
        for (i = 0; i < npct; i++) {
            char label[32];
            pct_label(pcts[i], label, sizeof label);
            printf(",%s", label);
        }
        printf("\n");
        printf("%zu," NUM_FMT "," NUM_FMT "," NUM_FMT "," NUM_FMT,
               n, a[0], a[n - 1], mean, median);
        for (i = 0; i < npct; i++)
            printf("," NUM_FMT, nearest ? pct_nearest(a, n, pcts[i])
                                        : pct_linear(a, n, pcts[i]));
        printf("\n");
    } else {
        printf("count      %zu\n", n);
        printf("min        " NUM_FMT "\n", a[0]);
        printf("max        " NUM_FMT "\n", a[n - 1]);
        printf("mean       " NUM_FMT "\n", mean);
        printf("median     " NUM_FMT "\n", median);
        for (i = 0; i < npct; i++) {
            char label[32];
            pct_label(pcts[i], label, sizeof label);
            printf("%-10s " NUM_FMT "\n", label,
                   nearest ? pct_nearest(a, n, pcts[i])
                           : pct_linear(a, n, pcts[i]));
        }
        printf("method     %s\n", method);
    }

    free(a);
    return 0;
}
