// https://codereview.stackexchange.com/questions/105114/re-implementing-memcpy 
#include <stdio.h>
#include <string.h>
#include <time.h>

void *memcpy2(void *dst, const void *src,size_t n)
{
    size_t i;

    for (i=0;i<n;i++)
        *(char *) dst++ = *(char *) src++;
    return dst;
}

void *memcpy3(void *dst, const void *src, size_t n)
{
    void *ret = dst;
    asm volatile("rep movsb" : "+D" (dst) : "c"(n), "S"(src) : "cc", "memory");
    return ret;
}

int main(int argc, char *argv[])
{
    clock_t start;
    char dst[1024];
    char src[1024];

    start = clock(); for (int i = 0; i < 100000; ++i) memcpy(dst,src,1024);
    printf("memcpy time cost:%f ms\n", (double)(clock()-start)/CLOCKS_PER_SEC*1000);

    start = clock(); for (int i = 0; i < 100000; ++i) memcpy2(dst,src,1024);
    printf("memcpy2 time cost:%f ms\n", (double)(clock()-start)/CLOCKS_PER_SEC*1000);

    start = clock(); for (int i = 0; i < 100000; ++i) memcpy3(dst,src,1024);
    printf("memcpy3 time cost:%f ms\n", (double)(clock()-start)/CLOCKS_PER_SEC*1000);

    return 0;
}
