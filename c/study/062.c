#include <stdio.h>
int main(int argc, char *argv[])
{
    char* s = "a1b1c1", *p=s, lookback;
    do {
        if (lookback != 'a' && *p == '1') putchar(*p);
        lookback = *p;
    } while (*p++!='\0');
    return 0;
}
