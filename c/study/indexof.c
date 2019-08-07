#include <stdio.h>
#include <string.h>

static int indexof(char *str, char *substr) {
    int i, j;
    for (i = 0, j = 0; i < strlen(str); i++) {
        printf("debug:i=%d, j=%d, str[i]=%c, substr[j]=%c\n", i, j, str[i], substr[j]); 
        j = str[i] == substr[j] ? j + 1 : 0;
        if (j == strlen(substr)) return i - j + 1;
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
        NULL
    };

    for (p = tests; *p != NULL; p+=2) {
        str = *p;
        substr = *(p+1);
        ret = indexof(str, substr);
        printf("'%s'.indexof('%s')=%d\n", str, substr, ret);
    }
    return 0;
}
