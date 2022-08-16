//https://github.com/brichard19/memcpy_sse
// gcc -O3 060.c && ./a.out 
#include<stdio.h>
#include<string.h>
#include<malloc.h>

#include "emmintrin.h"

#ifdef _WIN32
#include<Windows.h>
unsigned int getTime()
{
    return GetTickCount();
}

void *malloc_aligned(size_t size)
{
    return _aligned_malloc(size, 16);
}

void free_aligned(void *p)
{
    _aligned_free(p);
}
#else

#include<sys/time.h>
#include<stdlib.h>

unsigned int getTime()
{
    struct timeval t;
    gettimeofday( &t, NULL );
    return t.tv_sec * 1000 + t.tv_usec / 1000;
}

void *malloc_aligned(size_t size)
{
    void *tmp;
    if(posix_memalign(&tmp, 16, size)) {
        return NULL;
    }

    return tmp;
}

void free_aligned(void *p)
{
    free(p);
}
#endif


#pragma GCC diagnostic ignored "-Wpointer-arith"
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

void memcpy4(void *dest, const void *src, size_t count)
{
	__m128i *srcPtr = (__m128i *)src;
	__m128i *destPtr = (__m128i *)dest;

	unsigned int index = 0;
	while(count) {

		__m128i x = _mm_load_si128(&srcPtr[index]);
		_mm_stream_si128(&destPtr[index], x);

		count -= 16;
		index++;
	}
}

void run_test(void *dest, void *src, size_t count, int iterations, int type)
{
	if(type==1) {
		for(int i = 0; i < iterations; i++) {
			memcpy(dest, src, count);
		}
	} else if (type==2){
		for(int i = 0; i < iterations; i++) {
			memcpy2(dest, src, count);
		}
	} else if (type==3){
		for(int i = 0; i < iterations; i++) {
			memcpy3(dest, src, count);
		}
	} else if (type==4){
		for(int i = 0; i < iterations; i++) {
			memcpy4(dest, src, count);
		}
	}


}

int main(int argc, char **argv)
{
    const unsigned int BUFFER_SIZE = 64 * 1024 * 1024;
	int iterations = 50;
	unsigned char *buf1 = (unsigned char *)malloc_aligned(BUFFER_SIZE);
	unsigned char *buf2 = (unsigned char *)malloc_aligned(BUFFER_SIZE);
	
	unsigned int t0, t1;
    for (int i = 1; i <= 4; ++i) {
        printf("Running memcpy%d\n", i);
        t0 = getTime();
        run_test(buf1, buf2, BUFFER_SIZE, iterations, i);
        t1 = getTime() - t0;

        printf("%.3fs\n", (float)t1/1000);
    }

	free_aligned(buf1);
	free_aligned(buf2);

	return 0;
}
