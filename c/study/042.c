#include <time.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>

int main(int argc, char *argv[])
{
    time_t start1, end1;
    start1 = time(NULL);
    for (int i = 0; i < 300000000; ++i) sqrt(i);
    end1 = time(NULL);
    printf("time=%f\n", difftime(end1, start1));

    clock_t start2, end2;
    start2 = clock();
    for (int i = 0; i < 300000000; ++i) sqrt(i);
    end2 = clock();
    printf("time2=%f\n",(double)(end2-start2)/CLOCKS_PER_SEC);

    struct timeval start3, end3;
    gettimeofday(&start3, NULL );
    for (int i = 0; i < 300000000; ++i) sqrt(i);
    gettimeofday(&end3, NULL );
    long timeuse =1000000 * (end3.tv_sec - start3.tv_sec ) + end3.tv_usec - start3.tv_usec;
    printf("time3=%f\n",timeuse /1000000.0);

    return 0;
}

