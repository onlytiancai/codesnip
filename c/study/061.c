#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <semaphore.h>


const int MAX = 100000;
const int THREAD_COUNT = 2;
int number = 0;
sem_t sem;

void *myfunc(void *arg) {
    for (int i = 0; i < MAX; i++) {
        sem_wait(&sem);
        number++;
        sem_post(&sem);
    }
    pthread_exit(&number);
}

int main(void) {
    sem_init(&sem, 0, 1);
    pthread_t threads[THREAD_COUNT];
    for (int i = 0; i < THREAD_COUNT; i++) {
        pthread_create(&threads[i], NULL, myfunc, NULL);
    }

    for (int i = 0; i < THREAD_COUNT; i++) {
        pthread_join(threads[i], NULL);
    }
    printf("number = %d\n", number);

    sem_destroy(&sem);
    return 0;
}
