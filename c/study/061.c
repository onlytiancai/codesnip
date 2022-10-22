#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <semaphore.h>
#include <time.h>

# define destroy_sem_scope __attribute__((cleanup(destroy_sem)))

const int MAX = 100000;
const int THREAD_COUNT = 2;
int number = 0;

void destroy_sem(sem_t *p_sem) {
    printf("destroy sem:%p\n", p_sem);
    sem_destroy(p_sem);
}

void *myfunc1(void *arg) {
    sem_t *p_sem = arg;
    printf("child thread get sem:%lu, %p\n", pthread_self(), p_sem);

    for (int i = 0; i < MAX; i++) {
        sem_wait(p_sem);
        number++;
        sem_post(p_sem);
    }

    pthread_exit(&number);
}

void test1() {
    number = 0;
    sem_t sem destroy_sem_scope;
    sem_init(&sem, 0, 1);
    printf("init sem:%p\n", &sem);

    clock_t start, end;

    pthread_t threads[THREAD_COUNT];
    for (int i = 0; i < THREAD_COUNT; i++) {
        pthread_create(&threads[i], NULL, myfunc1, (void*)&sem);
    }

    start = clock();
    for (int i = 0; i < THREAD_COUNT; i++) {
        pthread_join(threads[i], NULL);
    }
    printf("test1: number=%d, time cost=%fms\n", number, (double)(clock()-start)/CLOCKS_PER_SEC*1000);

}

void *myfunc2(void *arg) {
    sem_t *p_sem = arg;
    printf("child thread get sem:%lu, %p\n", pthread_self(), p_sem);

    sem_wait(p_sem);
    for (int i = 0; i < MAX; i++) {
        number++;
    }
    sem_post(p_sem);

    pthread_exit(&number);
}

void test2() {
    number = 0;
    sem_t sem destroy_sem_scope;
    sem_init(&sem, 0, 1);
    printf("init sem:%p\n", &sem);

    clock_t start, end;

    pthread_t threads[THREAD_COUNT];
    for (int i = 0; i < THREAD_COUNT; i++) {
        pthread_create(&threads[i], NULL, myfunc2, (void*)&sem);
    }

    start = clock();
    for (int i = 0; i < THREAD_COUNT; i++) {
        pthread_join(threads[i], NULL);
    }
    printf("test1: number=%d, time cost=%fms\n", number, (double)(clock()-start)/CLOCKS_PER_SEC*1000);

}
int main(void) {
    test1();
    test2();
    return 0;
}
