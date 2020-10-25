// 给你一个字符串 s 和一个字符规律 p，
// 请你来实现一个支持 '.' 和 '*' 的正则表达式匹配。
#include <stdio.h>

typedef int bool;
#define true 1
#define false 0 


bool isMatch(char * s, char * p) {
    bool star = 0;
    char c;
    while(c=*p++) {
        if (star = *p == '*')p++;
        if (!star) {
            if (*s++ != c && c != '.') return false;
        } else {
            if (c != '.') while(*s++ == c) ;
            else while(*s++ != *p) ;
            s--;
        }
    }
    return *s == '\0';
}

int main(int argc, char *argv[]) {
    printf("actual %d expect %d\n", isMatch("aa", "a"), false); 
    printf("actual %d expect %d\n", isMatch("aa", "a*"), true); 
    printf("actual %d expect %d\n", isMatch("ab", ".*"), true); 
    printf("actual %d expect %d\n", isMatch("aab", "c*a*b"), true); 
    printf("actual %d expect %d\n", isMatch("mississippi", "mis*is*p*."), false); 
    printf("actual %d expect %d\n", isMatch("ab", ".*c"), false); 
    return 0;
}
