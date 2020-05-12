#include <stdio.h>
#include <string.h>

#define MAX 1024

static char* trim(char *s1) {
    static char s2[MAX];
    bzero(s2, MAX);

    char *pre = "http://127.0.0.1";
    int prelen = strlen(pre);

    // 前缀填充
    strcpy(s2, pre);

    // 遍历每个字符
    int last = 0, i, j, pos = prelen;
    for (i = 0; i < strlen(s1); i++) {
        // 找到 / 或字符串末尾，作为一个段
        if (s1[i] == '/' || i == strlen(s1) - 1) {
            // 找到非 .. 段，直接复制到目标缓冲区
            if (strncmp("..", &s1[last], 2) != 0) {
                // 目标缓冲区末端补 /
                if (pos == prelen || s2[pos-1] != '/')
                    s2[pos++] = '/';

                for (j = last; j <= i ; j++) {
                    if (s1[j] != '/') s2[pos++] = s1[j];
                }
            } else { // 找到 .. 段，目标缓冲区向前回溯一个段
                // 先去掉末尾 / 再回溯
                if (s2[pos-1] == '/') pos = pos -1;
                while (pos > prelen) {
                    s2[pos--] = '\0';
                    if (s2[pos-1] == '/') break;
                }
            }
            //printf("debug:%.*s\n\t%s\n", i-last, &s1[last], s2);
            last = i + 1;
        }
    }
    return s2;
}

int main() {
    char *inputs[] = {
        "/index.m3u8", "http://127.0.0.1/index.m3u8",
        "/test/../../index0.ts", "http://127.0.0.1/index0.ts",
        "/test/../helloworld/../index1.ts", "http://127.0.0.1/index1.ts",
        "/test/../../helloworld/xiaohei/../index2.ts", "http://127.0.0.1/helloworld/index2.ts",
        "/test/helloworld/xiaohei/../index3.ts", "http://127.0.0.1/test/helloworld/index3.ts",
        "/../test/helloworld/xiaohei/../index4.ts", "http://127.0.0.1/test/helloworld/index4.ts",
        "/./test/helloworld/xiaohei/../index5.ts", "http://127.0.0.1/./test/helloworld/index5.ts",
        "/test/helloworld/xiaohei/../index6.ts", "http://127.0.0.1/test/helloworld/index6.ts",
        "/../../test/helloworld/xiaohei/../../../../../../index7.ts", "http://127.0.0.1/index7.ts"
    };

    int i;
    char *actual, *expect;
    for (i = 0; i < 18; i+=2) {
        expect = inputs[i+1];
        actual = trim(inputs[i]);
        if (strncmp(expect, actual, MAX) == 0) {
            printf("test: %d success\n", i / 2 + 1);
        } else {
            printf("test: %d failed\n", i / 2 + 1);
            printf("\t input : %s\n", inputs[i]);
            printf("\t expect: %s\n", expect);
            printf("\t actual: %s\n", actual);
        }
    }
    return 0;
}
