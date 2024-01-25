#include <stdio.h>
#include <stdint.h>

int main()

{
        uintptr_t a = 12345;
        uintptr_t *p = &a;
        uintptr_t ptr = (uintptr_t )p;
        printf("%lx\n",ptr);
        printf("sizeof(ptr):%ld,sizeof(p):%ld\n",sizeof(ptr),sizeof(p));
        return 0;
}
