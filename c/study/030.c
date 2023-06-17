#include<stdio.h>  
#include<pthread.h> 
#include <sys/time.h>

pthread_mutex_t lock;
int g_data = 0;
const int N = 10000000;

void* thread1(void *data)  {
    int i;
    float startTime = (float)clock()/CLOCKS_PER_SEC, endTime;

    for (i = 0; i < N; ++i) {
        pthread_mutex_lock(&lock); 
        g_data++;
        pthread_mutex_unlock(&lock);  
    }

    endTime = (float)clock()/CLOCKS_PER_SEC;
    printf("%s:%d:%f\n", (char*)data, N, endTime - startTime);
    pthread_exit(NULL);  
    return 0;
}

void* thread2(void *data)  {
    int i;
    float startTime = (float)clock()/CLOCKS_PER_SEC, endTime;

    for (i = 0; i < N; ++i) {
        pthread_mutex_lock(&lock); 
        g_data--;
        pthread_mutex_unlock(&lock);  
    }

    endTime = (float)clock()/CLOCKS_PER_SEC;
    printf("%s:%d:%f\n", (char*)data, N, endTime - startTime);
    pthread_exit(NULL);  
    return 0;
}

int main(){
    pthread_t ptid1,ptid2; 

    pthread_mutex_init(&lock,NULL);
    pthread_create(&ptid1, NULL, thread1, "thread1:increase");
    pthread_create(&ptid2, NULL, thread2, "thread2:decrease");

    pthread_join(ptid1, NULL);
    pthread_join(ptid2, NULL);
    pthread_mutex_destroy(&lock);

    printf("g_data=%d\n", g_data);
    return 0;
}
