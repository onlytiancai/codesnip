/*
 * test_stats.c —— stats.c 核心函数的单元测试（零外部依赖）
 *
 * 编译: cc -std=c11 -O2 -Wall -Wextra -pedantic -o tests/test_stats tests/test_stats.c -lm
 * 运行: ./tests/test_stats
 *
 * 生产代码保持单文件自包含，这里用 #include "../stats.c" 直接复用被测的
 * static 函数：先把 main 改名，避免与测试的 main 冲突。
 */
#define main stats_cli_main
#include "../stats.c"
#undef main

static int checks = 0;
static int failures = 0;

static int close_enough(double got, double want)
{
    return isfinite(got) && fabs(got - want) <= 1e-9 * (1.0 + fabs(want));
}

#define OK_TRUE(label, cond) do {                                              \
    checks++;                                                                  \
    if (!(cond)) {                                                             \
        failures++;                                                            \
        fprintf(stderr, "  FAIL %-36s (line %d)\n", (label), __LINE__);        \
    }                                                                          \
} while (0)

#define OK_INT(label, got, want) do {                                          \
    long long g_ = (long long)(got), w_ = (long long)(want);                   \
    checks++;                                                                  \
    if (g_ != w_) {                                                            \
        failures++;                                                            \
        fprintf(stderr, "  FAIL %-36s got=%lld want=%lld (line %d)\n",         \
                (label), g_, w_, __LINE__);                                     \
    }                                                                          \
} while (0)

#define OK_DBL(label, got, want) do {                                          \
    double g_ = (double)(got), w_ = (double)(want);                            \
    checks++;                                                                  \
    if (!close_enough(g_, w_)) {                                               \
        failures++;                                                            \
        fprintf(stderr, "  FAIL %-36s got=%.17g want=%.17g (line %d)\n",       \
                (label), g_, w_, __LINE__);                                     \
    }                                                                          \
} while (0)

#define OK_STR(label, got, want) do {                                          \
    const char *g_ = (got), *w_ = (want);                                      \
    checks++;                                                                  \
    if (strcmp(g_, w_) != 0) {                                                 \
        failures++;                                                            \
        fprintf(stderr, "  FAIL %-36s got='%s' want='%s' (line %d)\n",         \
                (label), g_, w_, __LINE__);                                     \
    }                                                                          \
} while (0)

#define RUN(fn) do {                                                           \
    int before_ = failures;                                                    \
    fn();                                                                      \
    printf("  %-24s %s\n", #fn, failures == before_ ? "ok" : "FAIL");          \
} while (0)

static void test_parse_number(void)
{
    double x = 0.0;

    OK_INT("整数", parse_number("42", &x), 0);
    OK_DBL("整数值", x, 42.0);
    OK_INT("负数", parse_number("-3.5", &x), 0);
    OK_DBL("负数值", x, -3.5);
    OK_INT("科学计数法", parse_number("1e2", &x), 0);
    OK_DBL("科学计数法值", x, 100.0);
    OK_INT("前导加号", parse_number("+0.25", &x), 0);
    OK_DBL("前导加号值", x, 0.25);
    OK_INT("尾随非法字符", parse_number("12abc", &x), -1);
    OK_INT("空串", parse_number("", &x), -1);
    OK_INT("纯空白", parse_number(" ", &x), -1);
    OK_INT("nan 被拒绝", parse_number("nan", &x), -2);
    OK_INT("inf 被拒绝", parse_number("inf", &x), -2);
    OK_INT("1e400 溢出被拒绝", parse_number("1e400", &x), -2);
}

static void test_normalize_escapes(void)
{
    char b[64];

    strcpy(b, "100\\n50\\n22\\n");
    normalize_escapes(b);
    OK_STR("字面 \\n 变空白", b, "100 50 22 ");

    strcpy(b, "100\\n50\n22\\t46");
    normalize_escapes(b);
    OK_STR("字面转义与真实换行混合", b, "100 50\n22 46");

    strcpy(b, "1 2 3\n");
    normalize_escapes(b);
    OK_STR("普通行不被改动", b, "1 2 3\n");
}

static void test_parse_limit(void)
{
    OK_INT("上限 3", (long long)parse_limit("3"), 3);
    OK_INT("上限 1000（CLI 边界测试用）", (long long)parse_limit("1000"), 1000);
    OK_INT("上限 0 表示不限制", (long long)parse_limit("0"), 0);
    OK_INT("上限 1000000", (long long)parse_limit("1000000"), 1000000);
}

static void test_sort(void)
{
    double a[] = { 3.0, -1.0, 2.0, -1.0, 0.0 };

    qsort(a, 5, sizeof a[0], cmp_double);
    OK_DBL("排序 [0]", a[0], -1.0);
    OK_DBL("排序 [1]", a[1], -1.0);
    OK_DBL("排序 [2]", a[2], 0.0);
    OK_DBL("排序 [3]", a[3], 2.0);
    OK_DBL("排序 [4]", a[4], 3.0);
    OK_TRUE("比较器升序", cmp_double(&a[0], &a[4]) < 0);
}

static void test_pct_linear(void)
{
    double s[100];
    double one[] = { 42.0 };
    double ex[] = { -1.7e308, 1.7e308 };
    size_t i;

    for (i = 0; i < 100; i++) s[i] = (double)(i + 1);
    OK_DBL("1..100 p0", pct_linear(s, 100, 0), 1.0);
    OK_DBL("1..100 p50", pct_linear(s, 100, 50), 50.5);
    OK_DBL("1..100 p85", pct_linear(s, 100, 85), 85.15);
    OK_DBL("1..100 p90", pct_linear(s, 100, 90), 90.1);
    OK_DBL("1..100 p95", pct_linear(s, 100, 95), 95.05);
    OK_DBL("1..100 p99", pct_linear(s, 100, 99), 99.01);
    OK_DBL("1..100 p99.9", pct_linear(s, 100, 99.9), 99.901);
    OK_DBL("1..100 p100", pct_linear(s, 100, 100), 100.0);
    OK_DBL("单元素任意分位", pct_linear(one, 1, 99), 42.0);
    OK_DBL("异号极值 p85 不溢出", pct_linear(ex, 2, 85), 1.19e308);
    OK_TRUE("异号极值结果有限", isfinite(pct_linear(ex, 2, 85)));
}

static void test_pct_nearest(void)
{
    double s[100];
    double s7[] = { 1, 2, 3, 4, 5, 6, 7 };
    size_t i;

    for (i = 0; i < 100; i++) s[i] = (double)(i + 1);
    OK_DBL("1..100 最近秩 p0", pct_nearest(s, 100, 0), 1.0);
    OK_DBL("1..100 最近秩 p50", pct_nearest(s, 100, 50), 50.0);
    OK_DBL("1..100 最近秩 p85 浮点容差", pct_nearest(s, 100, 85), 85.0);
    OK_DBL("1..100 最近秩 p90", pct_nearest(s, 100, 90), 90.0);
    OK_DBL("1..100 最近秩 p95", pct_nearest(s, 100, 95), 95.0);
    OK_DBL("1..100 最近秩 p99", pct_nearest(s, 100, 99), 99.0);
    OK_DBL("1..100 最近秩 p100", pct_nearest(s, 100, 100), 100.0);
    OK_DBL("n=7 最近秩 p50", pct_nearest(s7, 7, 50), 4.0);
    OK_DBL("n=7 最近秩 p85", pct_nearest(s7, 7, 85), 6.0);
}

static void test_mean_of(void)
{
    double a[] = { 1, 2, 3, 4, 5 };
    double b[] = { 1e308, 1.7e308 };
    double c[] = { -1.7e308, 1.7e308 };
    double z[] = { 0, 0, 0 };
    double tenth[1000];
    size_t i;

    OK_DBL("基础均值", mean_of(a, 5), 3.0);
    OK_DBL("偶数个均值", mean_of(a, 4), 2.5);
    OK_DBL("全零均值", mean_of(z, 3), 0.0);
    OK_DBL("正负相消", mean_of(c, 2), 0.0);
    OK_DBL("极值均值不溢出", mean_of(b, 2), 1.35e308);
    OK_TRUE("极值均值有限", isfinite(mean_of(b, 2)));
    for (i = 0; i < 1000; i++) tenth[i] = 0.1;
    OK_DBL("1000 个 0.1 的均值精度", mean_of(tenth, 1000), 0.1);
}

static void test_median_of(void)
{
    double a[] = { 1, 2, 3, 4, 5 };
    double s[100];
    double b[] = { 1e308, 1.7e308 };
    double one[] = { 42.0 };
    size_t i;

    for (i = 0; i < 100; i++) s[i] = (double)(i + 1);
    OK_DBL("奇数个中位数", median_of(a, 5), 3.0);
    OK_DBL("偶数个中位数", median_of(a, 4), 2.5);
    OK_DBL("单元素中位数", median_of(one, 1), 42.0);
    OK_DBL("1..100 中位数", median_of(s, 100), 50.5);
    OK_DBL("中位数与 linear p50 一致", median_of(s, 100), pct_linear(s, 100, 50));
    OK_DBL("极值中位数不溢出", median_of(b, 2), 1.35e308);
    OK_TRUE("极值中位数有限", isfinite(median_of(b, 2)));
}

static void test_pct_label(void)
{
    char buf[32];

    pct_label(85, buf, sizeof buf);
    OK_STR("标签 85", buf, "p85");
    pct_label(99.9, buf, sizeof buf);
    OK_STR("标签 99.9", buf, "p99.9");
    pct_label(100, buf, sizeof buf);
    OK_STR("标签 100", buf, "p100");
}

static void test_parse_pct_list(void)
{
    double p[MAX_PERCENTILES];
    size_t n;

    n = parse_pct_list("85,90,95,99", p, MAX_PERCENTILES);
    OK_INT("默认列表个数", (long long)n, 4);
    OK_DBL("默认列表 [0]", p[0], 85.0);
    OK_DBL("默认列表 [3]", p[3], 99.0);

    n = parse_pct_list("99,99,90", p, MAX_PERCENTILES);
    OK_INT("去重后个数", (long long)n, 2);
    OK_DBL("去重后保持顺序 [0]", p[0], 99.0);
    OK_DBL("去重后保持顺序 [1]", p[1], 90.0);

    n = parse_pct_list(" 50 , 99.9 ", p, MAX_PERCENTILES);
    OK_INT("空白与小数分位个数", (long long)n, 2);
    OK_DBL("小数分位", p[1], 99.9);
}

static void test_vec_limit(void)
{
    Vec v = { NULL, 0, 0, 3 };
    Vec u = { NULL, 0, 0, 0 };
    size_t i;
    int all_ok = 1;

    OK_INT("默认上限为 100 万", (long long)DEFAULT_MAX_VALUES, 1000000);
    OK_INT("push #1", vec_push(&v, 1.0), 0);
    OK_INT("push #2", vec_push(&v, 2.0), 0);
    OK_INT("push #3 恰好到上限", vec_push(&v, 3.0), 0);
    OK_INT("push #4 超限返回 -2", vec_push(&v, 4.0), -2);
    OK_INT("超限后 len 不变", (long long)v.len, 3);
    OK_DBL("已有数据不被破坏", v.data[2], 3.0);
    free(v.data);

    for (i = 0; i < 300; i++)
        if (vec_push(&u, (double)i) != 0) all_ok = 0;
    OK_TRUE("max=0 时可连续 push 300 个", all_ok);
    OK_INT("动态扩容后 len", (long long)u.len, 300);
    OK_DBL("扩容后末元素", u.data[299], 299.0);
    free(u.data);
}

int main(void)
{
    puts("stats.c 核心函数单元测试");
    RUN(test_parse_number);
    RUN(test_normalize_escapes);
    RUN(test_parse_limit);
    RUN(test_sort);
    RUN(test_pct_linear);
    RUN(test_pct_nearest);
    RUN(test_mean_of);
    RUN(test_median_of);
    RUN(test_pct_label);
    RUN(test_parse_pct_list);
    RUN(test_vec_limit);
    printf("合计 %d 项断言，%d 项失败\n", checks, failures);
    return failures != 0;
}
