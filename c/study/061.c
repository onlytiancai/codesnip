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
    sem_destroy(p_sem);
}

void *myfunc1(void *arg) {
    sem_t *p_sem = arg;
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

    clock_t start;

    pthread_t threads[THREAD_COUNT];
    for (int i = 0; i < THREAD_COUNT; i++) {
        pthread_create(&threads[i], NULL, myfunc1, (void*)&sem);
    }

    start = clock();
    for (int i = 0; i < THREAD_COUNT; i++) {
        pthread_join(threads[i], NULL);
    }
    printf("test1(sem in the loop): number=%d, time cost=%fms\n", number, (double)(clock()-start)/CLOCKS_PER_SEC*1000);

}

void *myfunc2(void *arg) {
    sem_t *p_sem = arg;

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

    clock_t start;

    pthread_t threads[THREAD_COUNT];
    for (int i = 0; i < THREAD_COUNT; i++) {
        pthread_create(&threads[i], NULL, myfunc2, (void*)&sem);
    }

    start = clock();
    for (int i = 0; i < THREAD_COUNT; i++) {
        pthread_join(threads[i], NULL);
    }
    printf("test2(sem_t out the loop): number=%d, time cost=%fms\n", number, (double)(clock()-start)/CLOCKS_PER_SEC*1000);

}

void *myfunc3(void *arg) {
    pthread_spinlock_t *p_lock = arg;

    pthread_spin_lock(p_lock);
    for (int i = 0; i < MAX; i++) {
        number++;
    }
    pthread_spin_unlock(p_lock);

    pthread_exit(&number);
}

void test3() {
    number = 0;
    pthread_spinlock_t lock = 0;
    pthread_spin_init(&lock, PTHREAD_PROCESS_PRIVATE);

    clock_t start;

    pthread_t threads[THREAD_COUNT];
    for (int i = 0; i < THREAD_COUNT; i++) {
        pthread_create(&threads[i], NULL, myfunc3, (void*)&lock);
    }

    start = clock();
    for (int i = 0; i < THREAD_COUNT; i++) {
        pthread_join(threads[i], NULL);
    }
    pthread_spin_destroy(&lock);
    printf("test3(pthread_spinlock_t out the loop): number=%d, time cost=%fms\n", number, (double)(clock()-start)/CLOCKS_PER_SEC*1000);

}

void *myfunc4(void *arg) {
    pthread_mutex_t *p_lock = arg;

    pthread_mutex_lock(p_lock);
    for (int i = 0; i < MAX; i++) {
        number++;
    }
    pthread_mutex_unlock(p_lock);

    pthread_exit(&number);
}

void test4() {
    number = 0;

    pthread_mutex_t lock;
    pthread_mutex_init(&lock, NULL);

    clock_t start;

    pthread_t threads[THREAD_COUNT];
    for (int i = 0; i < THREAD_COUNT; i++) {
        pthread_create(&threads[i], NULL, myfunc4, (void*)&lock);
    }

    start = clock();
    for (int i = 0; i < THREAD_COUNT; i++) {
        pthread_join(threads[i], NULL);
    }
    pthread_mutex_destroy(&lock);
    printf("test4(pthread_mutex_t out the loop): number=%d, time cost=%fms\n", number, (double)(clock()-start)/CLOCKS_PER_SEC*1000);

}
int main(void) {
    test1();
    test2();
    test3();
    test4();
    return 0;
}
