#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <time.h>
#include <unistd.h>
#define _GNU_SOURCE
#define __USE_GNU
#include <sys/socket.h>
#include <fcntl.h>
#include <netdb.h>
#include<pthread.h>  
#include <sys/eventfd.h>
#include <sys/epoll.h>
#include <sys/types.h>
#include <sys/syscall.h>

const int N = 10;
const int MAX_EVENTS_SIZE = 10; 

static char* response = "HTTP/1.1 200 OK\r\nServer: nginx\r\nDate: Thu, 20 May 2021 04:16:43 GMT\r\nContent-Type: application/octet-stream\r\nContent-Length: 10\r\nConnection: close\r\nContent-Type: text/html;charset=utf-8\r\n\r\n127.0.0.1\n";

const int THREAD_COUNT = 30;
#define FD_QUEUE_MAX 100
struct ThreadData {
    pthread_t ptid; 
    int sfd;
    int epfd;
};

int guard(int n, char * err) { if (n == -1) { perror(err); exit(1); } return n; }


static int
create_and_bind (const char *port)
{
    struct addrinfo hints;
    struct addrinfo *result, *rp;
    int s, sfd;

    memset (&hints, 0, sizeof (struct addrinfo));
    hints.ai_family = AF_UNSPEC;     /* Return IPv4 and IPv6 choices */
    hints.ai_socktype = SOCK_STREAM; /* We want a TCP socket */
    hints.ai_flags = AI_PASSIVE;     /* All interfaces */

    s = getaddrinfo (NULL, port, &hints, &result);
    if (s != 0)
    {
        fprintf (stderr, "getaddrinfo: %s\n", gai_strerror (s));
        return -1;
    }

    int val =1;
    for (rp = result; rp != NULL; rp = rp->ai_next)
    {
        sfd = socket (rp->ai_family, rp->ai_socktype, rp->ai_protocol);
        if (sfd == -1)
            continue;
   
        if (setsockopt(sfd, SOL_SOCKET, SO_REUSEPORT, &val, sizeof(val))<0) {
            perror("setsockopt()");
            return -1;
        }

        s = bind (sfd, rp->ai_addr, rp->ai_addrlen);
        if (s == 0)
        {
            /* We managed to bind successfully! */
            printf("bind port success:%s\n", port);
            break;
        }

        close (sfd);
    }

    if (rp == NULL)
    {
        fprintf (stderr, "Could not bind\n");
        return -1;
    }

    freeaddrinfo (result);

    s = fcntl(sfd, F_SETFL, fcntl(sfd, F_GETFL, 0) | O_NONBLOCK);
    if (s== -1){
        perror("calling fcntl");
        return -1;
    }

    return sfd;
}

static void handle_error(char* file, int line) {
    printf("has error: file=\"%s\" line=%d error=\"%s\"\n", file, line, strerror(errno));
    exit(EXIT_FAILURE);
}

void* thread2(void *data) {
    struct ThreadData *td = (struct ThreadData*)data;
    int i, n, *p, nfds, socket_fd, s, j;
    struct epoll_event events[MAX_EVENTS_SIZE], event;
    //printf("worker[%ld]: thread start, lock=%p\n", syscall(__NR_gettid), &td->lock);
    
    event.data.fd = td->sfd;
    event.events = EPOLLIN | EPOLLET;
    guard(epoll_ctl(td->epfd, EPOLL_CTL_ADD, td->sfd, &event), "epoll_ctl error");

    while(1) {
        nfds = epoll_wait(td->epfd, events, MAX_EVENTS_SIZE, -1);
        //printf("worker[%ld]: epoll wait, nfds=%d\n", syscall(__NR_gettid), nfds);
        for (i = 0; i < nfds; i++) {
            //printf("worker[%ld]: foreach fd, fd=%d is_event_fd=%d\n", syscall(__NR_gettid), events[i].data.fd, events[i].data.fd == td->efd);
            if (td->sfd == events[i].data.fd) {

                if (!events[i].events & EPOLLIN) handle_error(__FILE__, __LINE__);
                
                /* We have a notification on the listening socket, which
                   means one or more incoming connections. */
                while (1)
                {
                    struct sockaddr in_addr;
                    socklen_t in_len;
                    int infd;

                    in_len = sizeof in_addr;
                    infd = accept4(td->sfd, &in_addr, &in_len, SOCK_NONBLOCK);
                    if (infd == -1)
                    {
                        /* We have processed all incoming connections. */
                        if (errno == EAGAIN || errno == EWOULDBLOCK)
                            break;
                        else
                            handle_error(__FILE__, __LINE__);
                    }
                    event.data.fd = infd;
                    event.events = EPOLLIN | EPOLLET;
                    guard(epoll_ctl(td->epfd, EPOLL_CTL_ADD, infd, &event), "epoll_ctl error");
                }

            } else {
                // socket fd
                int done = 0;

                while (1) {
                    ssize_t count;
                    char buf[512];

                    count = read(events[i].data.fd, buf, sizeof buf);
                    //printf("worker[%ld]: read request, fd=%d, read bytes=%ld\n", syscall(__NR_gettid), events[i].data.fd, count);
                    if (count == -1) {
                        /* If errno == EAGAIN, that means we have read all
                           data. So go back to the main loop. */
                        if (errno != EAGAIN) {
                            handle_error(__FILE__, __LINE__);
                        }
                        break;
                    }
                    else if (count == 0) {
                        /* End of file. The remote has closed the
                           connection. */
                        done = 1;
                        break;
                    }
                    

                    /* send response */
                    s = write (events[i].data.fd, response, strlen(response));
                    //printf ("worker[%ld]: send response, fd=%d, send bytes=%d, resp len=%ld\n", syscall(__NR_gettid), events[i].data.fd, s, strlen(response));
                    if (s == -1) {
                        perror ("response");
                        abort ();
                    } else if (s == strlen(response)) {
                        // 一次性发送完毕
                        //printf("worker[%ld]: close fd, fd=%d\n", syscall(__NR_gettid), events[i].data.fd); 
                        close (events[i].data.fd);
                        break;
                    } else {
                        handle_error(__FILE__, __LINE__);
                        //TODO: 未发送完毕 
                    }

                }

                if (done)
                {

                    //printf("worker[%ld]: close fd 2, fd=%d\n", syscall(__NR_gettid), events[i].data.fd); 
                    /* Closing the descriptor will make epoll remove it
                       from the set of descriptors which are monitored. */
                    close (events[i].data.fd);
                }

            }
        }
        
    }
}

int main()
{
    int sfd, s, i, epfd;
    const char *port = "8888";
    time_t t;
    struct ThreadData tds[THREAD_COUNT], *p_td;
    struct epoll_event event;

    srand((unsigned) time(&t));

    for (i = 0; i < THREAD_COUNT; ++i) {
        p_td = &tds[i];

        sfd = guard(create_and_bind(port), "create and bind error");
        guard(listen (sfd, SOMAXCONN), "lisetn error");
        p_td->sfd = sfd;

        epfd = guard(epoll_create1(EPOLL_CLOEXEC), "epoll_create error");
        p_td->epfd = epfd;

        pthread_create(&p_td->ptid, NULL, thread2, p_td);
    }

    for (i = 0; i < THREAD_COUNT; ++i) {
        struct ThreadData *p_td = &tds[i];
        pthread_join(p_td->ptid, NULL);
        close (p_td->sfd);
    }

    return EXIT_SUCCESS;
}
