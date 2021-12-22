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
#include <sys/sysinfo.h>

#define GCC_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)

#if (GCC_VERSION >= 40100)
/* 内存访问栅 */
  #define barrier()             	(__sync_synchronize())
/* 原子获取 */
  #define AO_GET(ptr)       		({ __typeof__(*(ptr)) volatile *_val = (ptr); barrier(); (*_val); })
/*原子设置，如果原值和新值不一样则设置*/
  #define AO_SET(ptr, value)        ((void)__sync_lock_test_and_set((ptr), (value)))
/* 原子交换，如果被设置，则返回旧值，否则返回设置值 */
  #define AO_SWAP(ptr, value)       ((__typeof__(*(ptr)))__sync_lock_test_and_set((ptr), (value)))
/* 原子比较交换，如果当前值等于旧值，则新值被设置，返回旧值，否则返回新值*/
  #define AO_CAS(ptr, comp, value)  ((__typeof__(*(ptr)))__sync_val_compare_and_swap((ptr), (comp), (value)))
/* 原子比较交换，如果当前值等于旧指，则新值被设置，返回真值，否则返回假 */
  #define AO_CASB(ptr, comp, value) (__sync_bool_compare_and_swap((ptr), (comp), (value)) != 0 ? true : false)
/* 原子清零 */
  #define AO_CLEAR(ptr)             ((void)__sync_lock_release((ptr)))
/* 通过值与旧值进行算术与位操作，返回新值 */
  #define AO_ADD_F(ptr, value)      ((__typeof__(*(ptr)))__sync_add_and_fetch((ptr), (value)))
  #define AO_SUB_F(ptr, value)      ((__typeof__(*(ptr)))__sync_sub_and_fetch((ptr), (value)))
  #define AO_OR_F(ptr, value)       ((__typeof__(*(ptr)))__sync_or_and_fetch((ptr), (value)))
  #define AO_AND_F(ptr, value)      ((__typeof__(*(ptr)))__sync_and_and_fetch((ptr), (value)))
  #define AO_XOR_F(ptr, value)      ((__typeof__(*(ptr)))__sync_xor_and_fetch((ptr), (value)))
/* 通过值与旧值进行算术与位操作，返回旧值 */
  #define AO_F_ADD(ptr, value)      ((__typeof__(*(ptr)))__sync_fetch_and_add((ptr), (value)))
  #define AO_F_SUB(ptr, value)      ((__typeof__(*(ptr)))__sync_fetch_and_sub((ptr), (value)))
  #define AO_F_OR(ptr, value)       ((__typeof__(*(ptr)))__sync_fetch_and_or((ptr), (value)))
  #define AO_F_AND(ptr, value)      ((__typeof__(*(ptr)))__sync_fetch_and_and((ptr), (value)))
  #define AO_F_XOR(ptr, value)      ((__typeof__(*(ptr)))__sync_fetch_and_xor((ptr), (value)))
#else
  #error "can not supported atomic operation by gcc(v4.0.0+) buildin function."
#endif	/* if (GCC_VERSION >= 40100) */
/* 忽略返回值，算术和位操作 */
#define AO_INC(ptr)                 ((void)AO_ADD_F((ptr), 1))
#define AO_DEC(ptr)                 ((void)AO_SUB_F((ptr), 1))
#define AO_ADD(ptr, val)            ((void)AO_ADD_F((ptr), (val)))
#define AO_SUB(ptr, val)            ((void)AO_SUB_F((ptr), (val)))
#define AO_OR(ptr, val)			 ((void)AO_OR_F((ptr), (val)))
#define AO_AND(ptr, val)			((void)AO_AND_F((ptr), (val)))
#define AO_XOR(ptr, val)			((void)AO_XOR_F((ptr), (val)))
/* 通过掩码，设置某个位为1，并返还新的值 */
#define AO_BIT_ON(ptr, mask)        AO_OR_F((ptr), (mask))
/* 通过掩码，设置某个位为0，并返还新的值 */
#define AO_BIT_OFF(ptr, mask)       AO_AND_F((ptr), ~(mask))
/* 通过掩码，交换某个位，1变0，0变1，并返还新的值 */
#define AO_BIT_XCHG(ptr, mask)      AO_XOR_F((ptr), (mask))

static int counter_bind = 0, counter_accept = 0, counter_epoll_wait = 0, counter_read = 0, counter_write = 0, counter_close = 0;
const int N = 10;
const int MAX_EVENTS_SIZE = 1024; 

const int MAX_RSP_LEN = 300;
static char* response = "HTTP/1.1 200 OK\r\nServer: wawa-server\r\nDate: Thu, 20 May 2021 04:16:43 GMT\r\nContent-Length: 10\r\nConnection: keep-alive\r\nContent-Type: text/html;charset=utf-8\r\n\r\n127.0.0.1\n";
static char *response_arr;

static int thread_count = 0;
#define FD_QUEUE_MAX 100
struct ThreadData {
    pthread_t ptid; 
    int sfd;
    int epfd;
};

int guard(int n, char * err) { if (n == -1) { perror(err); exit(1); } return n; }

int parse_req(char *str, int start, int end) {
    int c = -1, s = 0, i = 0, count = 0;
    char buf[1024], *p = buf;
    for (i=start; i < end;i++) {
        c = str[i];
        if (c == EOF) break;
        *p++ = c;

        if (s == 0 && c == '\r') s = 1; 
        else if (s == 1 && c == '\n') s = 2; 
        else if (s == 2 && c == '\r') s = 3; 
        else if (s == 3 && c == '\n') s = 4; 
        else s = 0; 
        
        if (s==4) {
            *(p-4) = '\0';
            //printf("===\n%s\n", buf);
            p = buf;
            count++;
        }
    }
    return count;
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
            AO_INC(&counter_bind);
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
    int i, n, *p, nfds, socket_fd, s, j, k;
    struct epoll_event events[MAX_EVENTS_SIZE], event;
    //printf("worker[%ld]: thread start, lock=%p\n", syscall(__NR_gettid), &td->lock);
    
    event.data.fd = td->sfd;
    event.events = EPOLLIN | EPOLLET;
    guard(epoll_ctl(td->epfd, EPOLL_CTL_ADD, td->sfd, &event), "epoll_ctl error");

    while(1) {
        nfds = epoll_wait(td->epfd, events, MAX_EVENTS_SIZE, -1);
        AO_INC(&counter_epoll_wait);
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
                    AO_INC(&counter_accept);
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
                    char buf[40000];

                    count = read(events[i].data.fd, buf, sizeof buf);
                    AO_INC(&counter_read);
                    //printf("worker[%ld]: read request, fd=%d, read bytes=%ld\n", syscall(__NR_gettid), events[i].data.fd, count);
                    if (count == -1) {
                        /* If errno == EAGAIN, that means we have read all
                           data. So go back to the main loop. */
                        if (errno != EAGAIN && errno != ECONNRESET) {
                            handle_error(__FILE__, __LINE__);
                        }
                        break;
                    }
                    else if (count == 0) {
                        /* End of file. The remote has closed the connection. */
                        close (events[i].data.fd);
                        AO_INC(&counter_close);
                        break;
                    }
                    
                    int req_count = MAX_RSP_LEN;
                    
                    //printf("parse_req count:%d\n", req_count);

                    /* send response */
                    s = write (events[i].data.fd, response_arr, strlen(response)*req_count);
                    AO_INC(&counter_write);
                    //printf ("worker[%ld]: send response, fd=%d, send bytes=%d, resp len=%ld\n", syscall(__NR_gettid), events[i].data.fd, s, strlen(response));
                    if (s == -1) {
                        handle_error(__FILE__, __LINE__);
                    } else if (s == strlen(response)*req_count) {
                        // 一次性发送完毕
                        //printf("worker[%ld]: close fd, fd=%d\n", syscall(__NR_gettid), events[i].data.fd); 
                    } else {
                        handle_error(__FILE__, __LINE__);
                        //TODO: 未发送完毕 
                    }

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
    struct epoll_event event;

    response_arr = (char*)malloc(strlen(response)*MAX_RSP_LEN+1);
    response_arr[strlen(response)*MAX_RSP_LEN] = '\0';
    for (i = 0; i < MAX_RSP_LEN; i++) {
        strncpy(response_arr+i*strlen(response), response, strlen(response)); 
    }

    thread_count = get_nprocs()-5;
    printf("thread_count=%d, port=%s.\n", thread_count, port);
    struct ThreadData tds[thread_count], *p_td;

    srand((unsigned) time(&t));

    for (i = 0; i < thread_count; ++i) {
        p_td = &tds[i];

        sfd = guard(create_and_bind(port), "create and bind error");
        guard(listen (sfd, SOMAXCONN), "lisetn error");
        p_td->sfd = sfd;

        epfd = guard(epoll_create1(EPOLL_CLOEXEC), "epoll_create error");
        p_td->epfd = epfd;

        pthread_create(&p_td->ptid, NULL, thread2, p_td);
    }
    
    int c_accept = 0, c_epoll_wait = 0, c_read = 0, c_write = 0, c_close = 0;
    i = 0;
    char buf[20];
    char time_buff[20];
    time_t timep;
    memset(time_buff, 0, 20);
    while(1) {
        time (&timep);
        int new_c_accept = AO_GET(&counter_accept),
            new_c_epoll_wait = AO_GET(&counter_epoll_wait),
            new_c_read = AO_GET(&counter_read),
            new_c_write = AO_GET(&counter_write),
            new_c_close = AO_GET(&counter_close);
        if (i % 10 == 0) {
            printf("%-10s%-8s%-20s%-20s%-20s%-20s%-20s\n","time", "bind", "accept", "epoll_wait", "read", "write", "close");
        }

        strncpy(time_buff, ctime(&timep)+11, 8);

        printf("%-10s%-8d", time_buff, AO_GET(&counter_bind));
        sprintf(buf, "%d(%d)", new_c_accept, (new_c_accept - c_accept));
        printf("%-20s", buf);
        sprintf(buf, "%d(%d)", new_c_epoll_wait, (new_c_epoll_wait - c_epoll_wait));
        printf("%-20s", buf);
        sprintf(buf, "%d(%d)", new_c_write, (new_c_write - c_write));
        printf("%-20s", buf);
        sprintf(buf, "%d(%d)", new_c_read, (new_c_read - c_read));
        printf("%-20s", buf);
        sprintf(buf, "%d(%d)", new_c_close, (new_c_close - c_close));
        printf("%-20s", buf);
        printf("\n");

        c_accept = new_c_accept;
        c_epoll_wait = new_c_epoll_wait;
        c_read = new_c_read;
        c_write = new_c_write;
        c_close = new_c_close;
        i++;
        sleep(1);
    }

    for (i = 0; i < thread_count; ++i) {
        struct ThreadData *p_td = &tds[i];
        pthread_join(p_td->ptid, NULL);
        close (p_td->sfd);
    }

    free(response_arr);
    return EXIT_SUCCESS;
}
