// 给你一个字符串 s 和一个字符规律 p，
// 请你来实现一个支持 '.' 和 '*' 的正则表达式匹配。
#include <stdio.h>

typedef int bool;
#define true 1
#define false 0 


bool isMatch(char * s, char * p) {
    bool star = 0;
    char c, *o_s = s;
    while(c=*p++) {
        printf("debug:%s %s\n", s, p-1);
        if (star = *p == '*')p++;
        if (!star) {
            s++; 
        } else {
            if (c != '.') {
                while(*s++ == c) ;
                if (s != o_s) s--;
                if (s != o_s && *p == c) s--; // aaa a*a 多吃了一个
                //if (s != o_s && p+1 && *(p+1) == '*') s--; // aa a*c*a 
            } else {
                while(*s != '\0' && *s++ != *p) ;
            }
        }
    }
    return *s == '\0';
}

int main(int argc, char *argv[]) {
    printf("actual %d expect %d\n", isMatch("aaa", "ab*a*c*a"), true); 
    printf("actual %d expect %d\n", isMatch("aa", "a*"), true); 
    printf("actual %d expect %d\n", isMatch("aaa", "a*a"), true); 
    printf("actual %d expect %d\n", isMatch("aa", "a"), false); 
    printf("actual %d expect %d\n", isMatch("ab", ".*"), true); 
    printf("actual %d expect %d\n", isMatch("aab", "c*a*b"), true); 
    printf("actual %d expect %d\n", isMatch("mississippi", "mis*is*p*."), false); 
    printf("actual %d expect %d\n", isMatch("ab", ".*c"), false); 
    return 0;
}
