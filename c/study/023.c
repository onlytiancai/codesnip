#include <stdio.h>

int main(int argc, char *argv[])
{
    int i;
    for (i = 0; i < 256; ++i) {
        printf("%d 里有 %d 个 1\n", i, __builtin_popcount(i)); 
    }
    return 0;
}
