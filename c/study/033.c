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

static char* response = "HTTP/1.1 200 OK\r\nServer: nginx\r\nDate: Thu, 20 May 2021 04:16:43 GMT\r\nContent-Type: application/octet-stream\r\nContent-Length: 9\r\nConnection: close\r\nContent-Type: text/html;charset=utf-8\r\n\r\n127.0.0.1";

const int THREAD_COUNT = 20;
#define FD_QUEUE_MAX 100
struct ThreadData {
    pthread_mutex_t lock;
    pthread_t ptid; 
    int efd;
    int epfd;
    int fd_queue[FD_QUEUE_MAX];
    int queue_len;
};
static int make_socket_non_blocking (int sfd)
{
    int flags, s;

    flags = fcntl (sfd, F_GETFL, 0);
    if (flags == -1)
    {
        perror ("fcntl");
        return -1;
    }

    flags |= O_NONBLOCK;
    s = fcntl (sfd, F_SETFL, flags);
    if (s == -1)
    {
        perror ("fcntl");
        return -1;
    }

    return 0;
}


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

    for (rp = result; rp != NULL; rp = rp->ai_next)
    {
        sfd = socket (rp->ai_family, rp->ai_socktype, rp->ai_protocol);
        if (sfd == -1)
            continue;

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
    uint64_t result;
    //printf("worker[%ld]: thread start, lock=%p\n", syscall(__NR_gettid), &td->lock);

    while(1) {
        nfds = epoll_wait(td->epfd, events, MAX_EVENTS_SIZE, -1);
        //printf("worker[%ld]: epoll wait, nfds=%d\n", syscall(__NR_gettid), nfds);
        for (i = 0; i < nfds; i++) {
            //printf("worker[%ld]: foreach fd, fd=%d is_event_fd=%d\n", syscall(__NR_gettid), events[i].data.fd, events[i].data.fd == td->efd);
            if (td->efd == events[i].data.fd) {
                // event fd
                if (events[i].events & EPOLLIN) {
                    read(events[i].data.fd, &result, sizeof(uint64_t));
                    pthread_mutex_lock(&td->lock); 
                    //printf("worker[%ld]: thread lock, fd=%d, lock=%p queue_len=%d\n", syscall(__NR_gettid), events[i].data.fd, &td->lock, td->queue_len);
                    for (j = 0; j < td->queue_len; ++j) {
                        socket_fd = td->fd_queue[j];
                        event.data.fd = socket_fd;
                        event.events = EPOLLIN | EPOLLET;
                        s = epoll_ctl (td->epfd, EPOLL_CTL_ADD, socket_fd, &event);
                    }
                    td->queue_len = 0;
                    pthread_mutex_unlock(&td->lock);  
                    //printf("worker[%ld]: thread unlock, fd=%d, lock=%p\n", syscall(__NR_gettid), events[i].data.fd, &td->lock);

                } else {
                    handle_error(__FILE__, __LINE__);
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
    int sfd, s, i;
    const char *port = "8888";

    time_t t;
    srand((unsigned) time(&t));

    int ret;

    sfd = create_and_bind(port);
    if (sfd == -1) handle_error(__FILE__, __LINE__);

    s = make_socket_non_blocking (sfd);
    if (s == -1) handle_error(__FILE__, __LINE__);

    s = listen (sfd, SOMAXCONN);
    if (s == -1) handle_error(__FILE__, __LINE__);

    struct ThreadData tds[THREAD_COUNT], *p_td;
    int temp_epfd, temp_efd;
    struct epoll_event temp_event;
    for (i = 0; i < THREAD_COUNT; ++i) {
        p_td = &tds[i];
        temp_efd = eventfd(0, EFD_NONBLOCK);
        if (temp_efd == -1) handle_error(__FILE__, __LINE__);

        temp_epfd = epoll_create1(EPOLL_CLOEXEC);
        if (temp_epfd == -1) handle_error(__FILE__, __LINE__);

        p_td->efd = temp_efd;
        p_td->epfd = temp_epfd;
        p_td->queue_len = 0;
        s = pthread_mutex_init(&p_td->lock, NULL);
        //printf("mutex init %d %d %p\n", i, s, &p_td->lock);
        if (s == -1) handle_error(__FILE__, __LINE__);

        temp_event.data.fd = p_td->efd;
        temp_event.events = EPOLLIN | EPOLLET;
        s = epoll_ctl(p_td->epfd, EPOLL_CTL_ADD, p_td->efd, &temp_event);
        if (s == -1) handle_error(__FILE__, __LINE__);

        pthread_create(&p_td->ptid, NULL, thread2, p_td);
    }

    uint64_t count = 1;

    int epfd = epoll_create1(0);
    if (epfd == -1) handle_error(__FILE__, __LINE__);

    struct epoll_event events[MAX_EVENTS_SIZE], event;
    event.data.fd = sfd;
    event.events = EPOLLIN | EPOLLET;
    s = epoll_ctl(epfd, EPOLL_CTL_ADD, sfd, &event);
    if (s == -1) handle_error(__FILE__, __LINE__);

    while (1)
    {
        int nfds, i;

        nfds = epoll_wait(epfd, events, MAX_EVENTS_SIZE, -1);
        //printf("main: epoll wait nfds=%d\n", nfds);
        for (i = 0; i < nfds; i++)
        {
            //printf("main: poll: fd=%d is_listen_fd=%d\n", events[i].data.fd, events[i].data.fd == sfd);
            if (sfd == events[i].data.fd)
            {
                /* We have a notification on the listening socket, which
                   means one or more incoming connections. */
                while (1)
                {
                    struct sockaddr in_addr;
                    socklen_t in_len;
                    int infd;
                    char hbuf[NI_MAXHOST], sbuf[NI_MAXSERV];

                    in_len = sizeof in_addr;
                    infd = accept4(sfd, &in_addr, &in_len, SOCK_NONBLOCK);
                    if (infd == -1)
                    {
                        if ((errno == EAGAIN) ||
                                (errno == EWOULDBLOCK))
                        {
                            /* We have processed all incoming
                               connections. */
                            break;
                        }
                        else
                        {
                            perror ("accept");
                            break;
                        }
                    }

                    /*
                    s = getnameinfo (&in_addr, in_len,
                            hbuf, sizeof hbuf,
                            sbuf, sizeof sbuf,
                            NI_NUMERICHOST | NI_NUMERICSERV);
                    if (s == 0)
                    {
                         printf("main: Accepted connection on descriptor %d "
                                "(host=%s, port=%s)\n", infd, hbuf, sbuf);
                    }

                    s = make_socket_non_blocking (infd);
                    if (s == -1)
                        abort ();
                    */
                    int n = rand() % THREAD_COUNT;
                    struct ThreadData *p_td = &tds[n];

                    pthread_mutex_lock(&p_td->lock); 
                    //printf("main: random thread, index=%d, lock=%p, queue_len=%d\n", n, &p_td->lock, p_td->queue_len);
                    if (p_td->queue_len >= FD_QUEUE_MAX) handle_error(__FILE__, __LINE__);
                    p_td->fd_queue[p_td->queue_len] = infd;
                    p_td->queue_len++; 
                    ret = write(p_td->efd, &count, sizeof(uint64_t));
                    pthread_mutex_unlock(&p_td->lock);  
                }
                continue;
            }
            else
            {
                //printf("main: no epid wait:%d nfds=%d sfd=%d\n", events[i].data.fd, nfds, sfd);
                handle_error(__FILE__, __LINE__);
            }
        }
    }


    for (i = 0; i < THREAD_COUNT; ++i) {
        struct ThreadData *p_td = &tds[i];
        pthread_join(p_td->ptid, NULL);
        pthread_mutex_destroy(&p_td->lock);
    }

    close (sfd);
    return EXIT_SUCCESS;
}
