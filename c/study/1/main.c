#include <stdio.h>

#include "hello.h"

int main(int argc, const char *argv[])
{
    int i;

    for (i = 0; i < argc; i++) {
        hello(argv[i]);
    }

    return 0;
}
