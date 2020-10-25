#include <stdio.h>

typedef int bool;
#define true 1
#define false 0 


bool isMatch(char * s, char * p) {
    bool star = 0;
    char c;
    while(c=*p++) {
        if (star = *p == '*')p++;
        // printf("debug: %c %d \n", c, star); 

        if (!star) { // a, .
            if (*s++ != c && c != '.') return false;
        } else {
            if (c != '.'){
                while(*s++ == c) ;
                s--;
            } else {
                while(*s++ != *p) ;
                s--;
            }
        }
    }
    // printf("debug1:%c %d\n", *s, *s=='\0');
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
