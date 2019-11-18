#include <stdio.h>
#include <string.h>

int main(int argc, char* argv[]) {
    printf("hello 008\n");
    
    char* str = "hello 00800";
    char* sub = "008";
    printf("str=%s,sub=%s\n", str, sub);

    int index = -1;
    char *p = str, *q=sub;
    while (*p) {
        printf("p=%c\n", *p);
        if (*p==*q) {
            printf("\tq=%c\n", *q);
            q++; 
            if (*q=='\0') {
                index = p - str - strlen(sub) + 1;
                printf("sub found.\n"); 
            }
        } else {
            q=sub; 
        }

        p++;
    }
    printf("index=%d\n", index);
    
    return 0;
}
