#include <stdio.h>
#include <stdlib.h>

int main(void)
{
    int ret_code = 0;
    for (char c = 'a'; ret_code != EOF && c != 'z'; c++)
    {
        putc(c, stdout); 
    }
    putc('\n', stdout);

    return EXIT_SUCCESS;
}
