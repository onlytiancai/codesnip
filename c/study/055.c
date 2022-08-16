#include <stdint.h>
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <stdio.h>
#include <string.h>

int join_path(char *buf, int buflen, const char* base, const char* file)
{
    if (buf == NULL || base == NULL || file == NULL) return -1;

    if (buflen -1 < strnlen(base, buflen)) return -2; 
    strncpy(buf, base, buflen);

    if (buflen -1 < strnlen(buf, buflen) +1) return -3;
    strcat(buf, "/");

    if (buflen -1 < strnlen(buf, buflen) + strlen(file)) return -4;
    strncat(buf, file, buflen);
    return 0;
}

static void test_join_path_01(void **state) {
    char buf[100];
    int ret = join_path(buf, 100, "/tmp", "abc.jpg");
    assert_int_equal(0, ret);
    assert_string_equal(buf, "/tmp/abc.jpg");
}

static void test_join_path_02(void **state) {
    int ret = join_path(NULL, 5, NULL, NULL);
    assert_int_equal(-1, ret);
}

static void test_join_path_03(void **state) {
    char buf[4];
    int ret = join_path(buf, 4, "/tmp", "abc.jpg");
    assert_int_equal(-2, ret);
}

static void test_join_path_04(void **state) {
    char buf[5];
    int ret = join_path(buf, 5, "/tmp", "abc.jpg");
    assert_int_equal(-3, ret);
}

static void test_join_path_05(void **state) {
    char buf[12];
    int ret = join_path(buf, 12, "/tmp", "abc.jpg");
    assert_int_equal(-4, ret);
}

int main(int argc, char *argv[])
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_join_path_01),
        cmocka_unit_test(test_join_path_02),
        cmocka_unit_test(test_join_path_03),
        cmocka_unit_test(test_join_path_04),
        cmocka_unit_test(test_join_path_05),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
