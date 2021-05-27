#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <time.h>
#include <unistd.h>
#include <sys/socket.h>
#include <fcntl.h>
#include <netdb.h>
#include<pthread.h>  
#include <sys/eventfd.h>
#include <sys/epoll.h>

#define MAXEVENTS 64
const int N = 10;

static char* response = "HTTP/1.1 200 OK\r\nServer: nginx\r\nDate: Thu, 20 May 2021 04:16:43 GMT\r\nContent-Type: application/octet-stream\r\nContent-Length: 9\r\nConnection: close\r\nContent-Type: text/html;charset=utf-8\r\n\r\n127.0.0.1";

const int THREAD_COUNT = 2;
#define FD_QUEUE_MAX 10
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
    int i, n, *p, nfds, socket_fd, s;
    const int MAX_EVENTS_SIZE = 1; 
    struct epoll_event events[MAX_EVENTS_SIZE], event;
    uint64_t result;

    while(1) {
        nfds = epoll_wait(td->epfd, events, MAX_EVENTS_SIZE, -1);
        for (i = 0; i < nfds; i++) {
            if (td->efd == events[i].data.fd) {
                // event fd
                if (events[i].events & EPOLLIN) {
                    read(events[i].data.fd, &result, sizeof(uint64_t));
                    pthread_mutex_lock(&td->lock); 
                    for (i = 0; i < td->queue_len; ++i) {
                        socket_fd = td->fd_queue[i];
                        event.data.fd = socket_fd;
                        event.events = EPOLLIN | EPOLLET;
                        s = epoll_ctl (td->epfd, EPOLL_CTL_ADD, socket_fd, &event);
                    }
                    td->queue_len = 0;
                    pthread_mutex_unlock(&td->lock);  

                } else {
                    handle_error(__FILE__, __LINE__);
                }
            } else {
                // socket fd
                int done = 0;

                while (1) {
                    ssize_t count;
                    char buf[512];

                    count = read (events[i].data.fd, buf, sizeof buf);
                    if (count == -1) {
                        /* If errno == EAGAIN, that means we have read all
                           data. So go back to the main loop. */
                        if (errno != EAGAIN) {
                            perror ("read");
                            done = 1;
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
                    // printf ("send response:%d %ld\n",s, strlen(response));
                    if (s == -1) {
                        perror ("response");
                        abort ();
                    } else if (s == strlen(response)) {
                        // 一次性发送完毕
                        close (events[i].data.fd);
                        break;
                    } else {
                        //TODO: 未发送完毕 
                    }

                }

                if (done)
                {
                    /*
                       printf ("Closed connection on descriptor %d\n",
                       events[i].data.fd);
                       */
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
    struct epoll_event event;
    struct epoll_event *events;
    int epfd = -1;

    time_t t;
    srand((unsigned) time(&t));

    epfd = epoll_create1(EPOLL_CLOEXEC);
    if (epfd == -1) handle_error(__FILE__, __LINE__);

    int ret;

    sfd = create_and_bind(port);
    if (sfd == -1) handle_error(__FILE__, __LINE__);

    s = make_socket_non_blocking (sfd);
    if (s == -1) handle_error(__FILE__, __LINE__);

    s = listen (sfd, SOMAXCONN);
    if (s == -1) handle_error(__FILE__, __LINE__);

    event.data.fd = sfd;
    event.events = EPOLLIN | EPOLLET;
    s = epoll_ctl (epfd, EPOLL_CTL_ADD, sfd, &event);
    if (s == -1) handle_error(__FILE__, __LINE__);

    struct ThreadData tds[THREAD_COUNT], td;
    int temp_epfd, temp_efd;
    struct epoll_event temp_event;
    for (i = 0; i < THREAD_COUNT; ++i) {
        td = tds[i];
        temp_efd = eventfd(0, EFD_NONBLOCK);
        if (temp_efd == -1) handle_error(__FILE__, __LINE__);

        temp_epfd = epoll_create1(EPOLL_CLOEXEC);
        if (temp_epfd == -1) handle_error(__FILE__, __LINE__);
        td.efd = temp_efd;
        td.epfd = temp_epfd;
        td.queue_len = 0;
        pthread_mutex_init(&td.lock, NULL);

        temp_event.data.fd = td.efd;
        temp_event.events = EPOLLIN | EPOLLET;
        s = epoll_ctl(td.epfd, EPOLL_CTL_ADD, td.efd, &event);
        if (s == -1) handle_error(__FILE__, __LINE__);

        pthread_create(&td.ptid, NULL, thread2, &td);
    }

    /* Buffer where events are returned */
    events = calloc (MAXEVENTS, sizeof event);
    uint64_t count = 1;

    while (1)
    {
        int n, i;

        n = epoll_wait(epfd, events, MAXEVENTS, -1);
        for (i = 0; i < n; i++)
        {
            if ((events[i].events & EPOLLERR) ||
                    (events[i].events & EPOLLHUP) ||
                    (!(events[i].events & EPOLLIN)))
            {
                /* An error has occured on this fd, or the socket is not
                   ready for reading (why were we notified then?) */
                fprintf (stderr, "epoll error\n");
                close (events[i].data.fd);
                continue;
            }

            else if (sfd == events[i].data.fd)
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
                    infd = accept (sfd, &in_addr, &in_len);
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

                    s = getnameinfo (&in_addr, in_len,
                            hbuf, sizeof hbuf,
                            sbuf, sizeof sbuf,
                            NI_NUMERICHOST | NI_NUMERICSERV);
                    if (s == 0)
                    {
                         printf("Accepted connection on descriptor %d "
                                "(host=%s, port=%s)\n", infd, hbuf, sbuf);
                    }

                    /* Make the incoming socket non-blocking and add it to the
                       list of fds to monitor. */
                    s = make_socket_non_blocking (infd);
                    if (s == -1)
                        abort ();

                    n = rand() % THREAD_COUNT + 1;
                    struct ThreadData td = tds[n];

                    pthread_mutex_lock(&td.lock); 
                    if (td.queue_len >= FD_QUEUE_MAX) handle_error(__FILE__, __LINE__);
                    td.fd_queue[td.queue_len] = infd;
                    td.queue_len++; 
                    ret = write(td.efd, &count, sizeof(uint64_t));
                    pthread_mutex_unlock(&td.lock);  
                }
                continue;
            }
            else
            {
                handle_error(__FILE__, __LINE__);
            }
        }
    }


    for (i = 0; i < THREAD_COUNT; ++i) {
        pthread_join(tds[i].ptid, NULL);
        pthread_mutex_destroy(&tds[i].lock);
    }

    free (events);
    close (sfd);
    return EXIT_SUCCESS;
}
