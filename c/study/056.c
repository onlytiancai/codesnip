#include <stdio.h>
#include <time.h>
#include <string.h>
#include <unistd.h>

void join01(char* buff, unsigned int buflen, const char *base, const char *file)
{
    snprintf(buff,buflen,"%s/%s", base, file);
}

void join02(char* buff, unsigned int buflen, const char *base, const char *file)
{
    strcpy(buff, base);
    strcat(buff, "/");
    strcat(buff, file);
}

void join03(char* buff, unsigned int buflen, const char *base, const char *file)
{
    char *p = buff;
    while (*base) *p++ = *base++;
    *p++ = '/';
    while (*file) *p++ = *file++;
    *p = '\0';
}

void join04(char* buff, unsigned int buflen, const char *base, const char *file)
{
    char *p = buff;
    while ((*p++ = *base++)) ;
    *(p-1) = '/';
    while ((*p++ = *file++)) ;
    *(p-1) = '\0';
}

void join05(char* buff, unsigned int buflen, const char *base, const char *file)
{
    char *p = buff;

    int base_len = strlen(base);
    size_t eight = base_len / 8;
    size_t single = base_len % 8;
    while(eight > 0){
        *p++ = *base++; *p++ = *base++; *p++ = *base++; *p++ = *base++;
        *p++ = *base++; *p++ = *base++; *p++ = *base++; *p++ = *base++;
        --eight;
    }
    while(single > 0){
        *p++ = *base++;
        --single;
    }
    *p++ = '/';

    base_len = strlen(file);
    eight = base_len / 8;
    single = base_len % 8;
    while(eight > 0){
        *p++ = *file++; *p++ = *file++; *p++ = *file++; *p++ = *file++;
        *p++ = *file++; *p++ = *file++; *p++ = *file++; *p++ = *file++;
        --eight;
    }
    while(single > 0){
        *p++ = *file++;
        --single;
    }
    *p++ = '\0';
}

int main(int argc, char *argv[])
{
    long l2_cache_size  = sysconf(_SC_LEVEL2_CACHE_SIZE);
    printf("l2 cache size: %ld\n", l2_cache_size);
    clock_t start;
    char buff[100], *base = "/home/ubuntu/src/codesnip/c/study", *file="abc.jpg";

    memset(buff, 0, 100);
    join01(buff, 100, base, file);
    printf("join01:%s\n", buff);
    memset(buff, 0, 100);
    join02(buff, 100, base, file);
    printf("join02:%s\n", buff);
    memset(buff, 0, 100);
    join03(buff, 100, base, file);
    printf("join03:%s\n", buff);
    memset(buff, 0, 100);
    join04(buff, 100, base, file);
    printf("join04:%s\n", buff);
    memset(buff, 0, 100);
    join05(buff, 100, base, file);
    printf("join05:%s\n", buff);

    start = clock(); 
    for (int i = 0; i < 100000; ++i) { join01(buff, 100, base, file); } 
    printf("join01:time cost=%fms\n",(double)(clock()-start)/CLOCKS_PER_SEC*1000);

    start = clock(); 
    for (int i = 0; i < 100000; ++i) { join02(buff, 100, base, file); } 
    printf("join02:time cost=%fms\n",(double)(clock()-start)/CLOCKS_PER_SEC*1000);

    start = clock(); 
    for (int i = 0; i < 100000; ++i) { join03(buff, 100, base, file); } 
    printf("join03:time cost=%fms\n",(double)(clock()-start)/CLOCKS_PER_SEC*1000);

    start = clock(); 
    for (int i = 0; i < 100000; ++i) { join04(buff, 100, base, file); } 
    printf("join04:time cost=%fms\n",(double)(clock()-start)/CLOCKS_PER_SEC*1000);

    start = clock(); 
    for (int i = 0; i < 100000; ++i) { join05(buff, 100, base, file); } 
    printf("join05:time cost=%fms\n",(double)(clock()-start)/CLOCKS_PER_SEC*1000);

    return 0;
}
