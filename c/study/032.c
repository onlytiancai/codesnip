#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <time.h>
#include <unistd.h>
#include<pthread.h>  
#include <sys/eventfd.h>
#include <sys/epoll.h>

pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
const int N = 10;
int has_data = 0, use_list = 1, data_len = 0, *g_list, *list1, *list2;
int epfd = -1;

void handle_error(char* file, int line) {
    printf("has error: file=\"%s\" line=%d error=\"%s\"\n", file, line, strerror(errno));
    exit(EXIT_FAILURE);
}

void* thread1(void *data)  {
    int i, n, *p, ret, efd;
    struct epoll_event event;

    while(1) {
        pthread_mutex_lock(&lock); 
        p = g_list;
        pthread_mutex_unlock(&lock);  
        
        n = rand() % N + 1;
        for (i = 0; i < n; ++i) { p[i] = rand(); }

        efd = eventfd(1, EFD_CLOEXEC|EFD_NONBLOCK);
        if (efd == -1) handle_error(__FILE__, __LINE__);
        event.data.fd = efd;
        event.events = EPOLLIN | EPOLLET;
        ret = epoll_ctl(epfd, EPOLL_CTL_ADD, efd, &event);
        if (ret != 0) handle_error(__FILE__, __LINE__);

        pthread_mutex_lock(&lock); 
        has_data = 1; data_len = n;

        ret = write(efd, (void*)0xffffffff, sizeof(uint64_t));
        pthread_mutex_unlock(&lock);  
        sleep(1);
    }
}

void* thread2(void *data) {
    int i, n, *p, nfds;
    const int MAX_EVENTS_SIZE = 1; 
    struct epoll_event events[MAX_EVENTS_SIZE];
    uint64_t result;

    while(1) {
        nfds = epoll_wait(epfd, events, MAX_EVENTS_SIZE, 5000);
        for (i = 0; i < nfds; i++) {
            if (events[i].events & EPOLLIN) {
                read(events[i].data.fd, &result, sizeof(uint64_t));
                close(events[i].data.fd);

                pthread_mutex_lock(&lock); 
                p = g_list; n = data_len; data_len = 0; has_data = 0;
                if (use_list == 1) { g_list = list2; use_list = 2; }
                else { g_list = list1; use_list = 1; }
                pthread_mutex_unlock(&lock);  

            } else {
                handle_error(__FILE__, __LINE__);
            }
        }
        
        printf("========use_list:%d, n:%d, nfds:%d\n", use_list, n, nfds);
        for (i = 0; i < n; ++i) {
            printf("%d\n", p[i]);     
        } 
    }
}

int main()
{
    pthread_t ptid1,ptid2; 
    int l1[N], l2[N];

    time_t t;
    srand((unsigned) time(&t));

    epfd = epoll_create1(EPOLL_CLOEXEC);
    if (epfd == -1) handle_error(__FILE__, __LINE__);

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
