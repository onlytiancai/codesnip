#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include<pthread.h>  

pthread_mutex_t lock;
const int N = 10;
int *g_list;

void* thread1(void *data)  {
    int list[N], i;
    while(1) {
        for (i = 0; i < N; ++i) {
            list[i] = rand();
        }

        pthread_mutex_lock(&lock); 
        g_list = list;
        pthread_mutex_unlock(&lock);  
        
        sleep(5);
    }
}

void* thread2(void *data) {
    int list[N], i, *p;

    while(1) {
        pthread_mutex_lock(&lock); 
        p = g_list;
        g_list = list;
        pthread_mutex_unlock(&lock);  

        printf("========\n");
        for (i = 0; i < N; ++i) {
            printf("%d\n", p[i]);     
        } 
        
        sleep(1);
    }
}

int main()
{
    pthread_t ptid1,ptid2; 
    int list[N];
    g_list = list;

    pthread_mutex_init(&lock,NULL);
    pthread_create(&ptid1, NULL, thread1, "thread1:increase");
    pthread_create(&ptid2, NULL, thread2, "thread2:decrease");

    pthread_join(ptid1, NULL);
    pthread_join(ptid2, NULL);
    pthread_mutex_destroy(&lock);

    return 0;
}
