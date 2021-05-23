/* 按 \r\n\r\n 分割字符流 */
#include <stdio.h>

int main()
{
    int c = -1, s = 0;
    char buf[100], *p = buf;
    while(1){
        c = getchar();
        if (c == EOF) break;
        *p++ = c;

        if (s == 0) {
            s = c == '\r' ? 1: 0;
        } else if (s == 1) {
            s = c == '\n' ? 2: 0;
        } else if (s == 2) {
            s = c == '\r' ? 3: 0;
        } else if (s == 3) {
            s = c == '\n' ? 4: 0;
        } else {
            s = 0;
        }
        
        if (s==4) {
            *(p-4) = '\0';
            printf("===\n%s\n", buf);
            p = buf;
        }
    }

    return 0;
}
