#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include<pthread.h>  
#include <time.h>

pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
const int N = 10;
int has_data = 0, use_list = 1, data_len = 0, *g_list, *list1, *list2;

void* thread1(void *data)  {
    int i, n, *p;
    while(1) {
        pthread_mutex_lock(&lock); 
        p = g_list;
        pthread_mutex_unlock(&lock);  
        
        n = rand() % N + 1;
        for (i = 0; i < n; ++i) { p[i] = rand(); }

        pthread_mutex_lock(&lock); 
        has_data = 1; data_len = n;
        pthread_cond_signal (&cond);
        pthread_mutex_unlock(&lock);  
        sleep(1);
    }
}

void* thread2(void *data) {
    int i, n, *p;

    while(1) {
        pthread_mutex_lock(&lock); 
        while (!has_data) pthread_cond_wait (&cond, &lock);
        p = g_list; n = data_len; data_len = 0; has_data = 0;
        if (use_list == 1) { g_list = list2; use_list = 2;}
        else {g_list = list1; use_list = 1;}
        pthread_mutex_unlock(&lock);  

        printf("========\n");
        for (i = 0; i < n; ++i) {
            printf("%d\n", p[i]);     
        } 
        
        sleep(1);
    }
}

int main()
{
    pthread_t ptid1,ptid2; 
    int l1[N], l2[N];

    time_t t;
    srand((unsigned) time(&t));

    list1 = l1; list2 = l2;
    g_list = list1; use_list = 1;

    pthread_mutex_init(&lock,NULL);
    pthread_create(&ptid1, NULL, thread1, "thread1:increase");
    pthread_create(&ptid2, NULL, thread2, "thread2:decrease");

    pthread_join(ptid1, NULL);
    pthread_join(ptid2, NULL);
    pthread_mutex_destroy(&lock);

    return 0;
}
