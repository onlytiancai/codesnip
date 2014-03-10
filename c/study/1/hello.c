#include <stdio.h>
#include "hello.h"

static int i = 0;

void hello(const char* name)
{
    printf("hello %d %s .", ++i, name);
    printf("%d\n", bar());
}
