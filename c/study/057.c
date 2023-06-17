// from 小明
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

void join01(char *buff, uint32_t buff_len, const char *dir, const char *file)
{
    char *p = buff;
    uint32_t len = strlen(dir);
    size_t eight = len / 8;
    size_t single = len % 8;

    while(eight > 0)
    {
        *p++ = *dir++;
        *p++ = *dir++;
        *p++ = *dir++;
        *p++ = *dir++;
        *p++ = *dir++;
        *p++ = *dir++;
        *p++ = *dir++;
        *p++ = *dir++;
        eight--;
    }
    while(single > 0)
    {
        *p++ = *dir++;
        single--;
    }
    *p++ = '/';

    len = strlen(file);
    eight = len / 8;
    single = len % 8;

    while(eight > 0)
    {
        *p++ = *file++;
        *p++ = *file++;
        *p++ = *file++;
        *p++ = *file++;
        *p++ = *file++;
        *p++ = *file++;
        *p++ = *file++;
        *p++ = *file++;
        eight--;
    }
    while(single > 0)
    {
        *p++ = *file++;
        single--;
    }
    *p++ = '\0';
}

void join02(char *buff, uint32_t buff_len, const char *dir, const char *file)
{
    char *p1, *p2;
    uint32_t len1 = strlen(dir);
    uint32_t len2 = strlen(file);
    size_t eight1 = len1 >> 3;
    size_t eight2 = len2 >> 3;
    size_t single1 = len1 & 0x0007;
    size_t single2 = len2 & 0x0007;

    p1 = buff;
    p2 = &buff[len1 + 1];
    buff[len1] = '/';
    buff[len1 + len2 + 1] = '\0';

    while((eight1 > 0)&&(eight2 > 0))
    {
        *p1++ = *dir++; *p2++ = *file++;
        *p1++ = *dir++; *p2++ = *file++;
        *p1++ = *dir++; *p2++ = *file++;
        *p1++ = *dir++; *p2++ = *file++;
        *p1++ = *dir++; *p2++ = *file++;
        *p1++ = *dir++; *p2++ = *file++;
        *p1++ = *dir++; *p2++ = *file++;
        *p1++ = *dir++; *p2++ = *file++;
        eight1--;
        eight2--;
    }
    while(eight1 > 0)
    {
        *p1++ = *dir++;
        *p1++ = *dir++;
        *p1++ = *dir++;
        *p1++ = *dir++;
        *p1++ = *dir++;
        *p1++ = *dir++;
        *p1++ = *dir++;
        *p1++ = *dir++;
        eight1--;
    }
    while(eight2 > 0)
    {
        *p2++ = *file++;
        *p2++ = *file++;
        *p2++ = *file++;
        *p2++ = *file++;
        *p2++ = *file++;
        *p2++ = *file++;
        *p2++ = *file++;
        *p2++ = *file++;
        eight2--;
    }

    while((single1 > 0)&&(single2 > 0))
    {
        *p1++ = *dir++; *p2++ = *file++;
        single1--;
        single2--;
    }
    while(single1 > 0)
    {
        *p1++ = *dir++;
        single1--;
    }
    while(single2 > 0)
    {
        *p2++ = *file++;
        single2--;
    }
}

int main(int argc, char *argv[])
{
    int i;
    clock_t start;
    char buff[100];

    join01(buff, 100, "/tmp/abc/123/456/789", "addfdsdsdfcd.jpg");
    printf("join01 result:%s\n", buff);
    join02(buff, 100, "/tmp/abc/123/456/789", "addfdsdsdfcd.jpg");
    printf("join02 result:%s\n", buff);

    start = clock();
    for(i = 0; i < 100000; i++)
    {
        join01(buff, 100, "/tmp/abc/123/456/789", "addfdsdsdfcd.jpg");
    }
    printf("join01:time cost = %fms\n", (double)(clock() - start) / CLOCKS_PER_SEC * 1000);

    start = clock();
    for(i = 0; i < 100000; i++)
    {
        join02(buff, 100, "/tmp/abc/123/456/789", "addfdsdsdfcd.jpg");
    }
    printf("join02:time cost = %fms\n", (double)(clock() - start) / CLOCKS_PER_SEC * 1000);

    return 0;
}
