#include <stdio.h>
#include <string.h>

static int indexof(const char *str, const char *substr) {
    int i, j;
    int lenstr = str == NULL ? 0 : strlen(str);
    int lensubstr = substr == NULL ? 0 : strlen(substr);

    for (i = 0, j = 0; i < lenstr; i++) {
        j = str[i] == substr[j] ? j + 1 : 0;
        if (j == lensubstr) return i - j + 1;
    }

    return -1;
}

int main(int argc, char **argv) {
    int i, ret;
    char *str, *substr, **p;
    char *tests[] = {
        "abc", "a",
        "abc", "b",
        "abc", "d",
        "abc", "ab",
        "abc", "bc",
        "a",   "abc",
        "大雄宝殿",   "宝殿",
        NULL
    };

    printf("NULL.indexof(NULL)=%d\n", indexof(NULL, NULL));
    for (p = tests; *p != NULL; p+=2) {
        str = *p;
        substr = *(p+1);
        ret = indexof(str, substr);
        printf("'%s'.indexof('%s')=%d\n", str, substr, ret);
    }
    return 0;
}
