#include <stdio.h>
#include <ctype.h>
#include <string.h>

int main(int argc, char* argv[]) {
    printf("hello 007\n");
    
    char* s = "73455";
    printf("s=%s\n", s);

    int len = strlen(s);
    printf("len=%d\n", len); 

    int i, n;
    n = 0;
    for (i = 0; i < len; i++) {
        printf("%c %d\n", s[i], s[i] - '0'); 
        n = n * 10 + s[i] - '0'; // 防止溢出
    }
    printf("n=%d\n", n); 
    
    return 0;
}
