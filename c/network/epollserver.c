/**
 * epoll 学习
 * server test:
 *  $ ./epollserver.o
 *      server start: 0.0.0.0:80
 *  $ ./epollserver.o 4000
 *      server start: 0.0.0.0:4000
 *  $ ./epollserver.o 0.0.0.0 4000
 *      server start: 0.0.0.0:4000
 *
 * client test:
 *      echo "123456789" | nc localhost 4000 -w 1
 * */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <sys/socket.h>
#include <sys/epoll.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>

#define DEFAULT_PORT 80
#define LISTENQ 1000
#define MAXLINE 5
#define MAXCONN 1000

static int epfd;
static char line[MAXLINE];
static int line_len;

struct myevent_s
{
    int used;
    struct sockaddr_in clientaddr;
} myevents[MAXCONN];

/**
 * 获取一个空闲事件
 * */
struct myevent_s * get_free_event()
{
    int i;
    for (i = 0; i < MAXCONN; i++) {
        if (!myevents[i].used) {
            printf("get_free_event:%d\n", i);
            myevents[i].used = 1;
            return &myevents[i];
        }
    }
    perror("free event notfound\n");
    exit(1);
}

/**
 * 释放一个事件回池
 * */
void release_event(struct myevent_s *ev)
{
    (*ev).used = 0;
}


/**
 * 从命令行参数获取监听的IP
 * */
char* get_host(int argc, char* argv[])
{
    if (argc < 3) return "0.0.0.0"; //TODO：这里的内存返回后会不会消失
    return argv[1];
}

/**
 * 从命令行参数获取监控的端口
 * */
int get_port(int argc, char* argv[])
{
    if (argc < 2) return DEFAULT_PORT;
    if (argc == 2) return (int) strtol(argv[1], (char **)NULL, 10);
    if (argc > 2) return (int) strtol(argv[2], (char **)NULL, 10);
    
}

/**
 * 添加监控套接字到epll
 * */
int add_listenfd_to_epoll(int listenfd, struct epoll_event *p_ev)
{
    struct epoll_event ev = *p_ev;
    ev.data.fd = listenfd;
    ev.events = EPOLLIN | EPOLLET;
    epoll_ctl(epfd, EPOLL_CTL_ADD, listenfd, &ev);
}

/**
 * 设置套接字为非阻塞模式
 * */
void setnonblocking(int sock)
{
    int opts;
    opts = fcntl(sock,F_GETFL);
    if(opts<0)
    {
        perror("fcntl(sock,GETFL)");
        exit(1);
    }
    opts = opts|O_NONBLOCK;
    if(fcntl(sock,F_SETFL,opts)<0)
    {
        perror("fcntl(sock,SETFL,opts)");
        exit(1);
    }   
}

/**
 * 根据ip, port创建要监听的套接字，bind，但先不listen
 * */
int create_listen_socket(char* host, int port)
{
    int listenfd = socket(AF_INET, SOCK_STREAM, 0);
    setnonblocking(listenfd);

    struct sockaddr_in serveraddr;
    bzero(&serveraddr, sizeof(serveraddr));
    serveraddr.sin_family = AF_INET;

    inet_aton(host, &(serveraddr.sin_addr));
    serveraddr.sin_port = htons(port);

    bind(listenfd, (struct sockaddr *)&serveraddr, sizeof(serveraddr));

    return listenfd;
}

/**
 * 处理监听套接字的事件，一般为产生新的客户端连接
 * */
int listenfd_event_handler(int listenfd)
{
    socklen_t clilen;
    //struct myevent_s *myevent = get_free_event();
    struct sockaddr_in clientaddr;

    int connfd = accept(listenfd, (struct sockaddr *)&clientaddr, &clilen);
    if (connfd < 0) {
        perror("connfd < 0");
        exit(1);
    }

    setnonblocking(connfd);

    char *str = inet_ntoa(clientaddr.sin_addr);
    printf("accapt a connection from %s:%d \n", str, clientaddr.sin_port);

    struct epoll_event ev;
    //ev.data.ptr = myevent;
    ev.data.fd = connfd;
    ev.events = EPOLLIN | EPOLLET;
    epoll_ctl(epfd, EPOLL_CTL_ADD, connfd, &ev);
}

int clientfd_revent_handler(struct epoll_event* event)
{
    printf("EPOLLIN\n");
    int sockfd;
    struct epoll_event ev = *event;
    //struct myevent_s *myevent = (struct myevent_s *)ev.data.ptr;
    //printf("myevent: %d", (*myevent).used);

    if ((sockfd = ev.data.fd) < 0) 
        return;

    if ((line_len = read(sockfd, line, MAXLINE)) < 0) {
        if (errno == ECONNRESET) {
            //release_event(myevent);
            close(sockfd);
            ev.data.fd = -1;
        } else {
            printf("readline error\n");
        }
    } else if (line_len == 0) {
        printf("client closed\n");
        //release_event(myevent);
        close(sockfd);
        ev.data.fd = -1;
    }

    line[line_len] = '\0';
    printf("read %s \n", line);

    ev.data.fd = sockfd;
    ev.events = EPOLLOUT | EPOLLET;
    epoll_ctl(epfd, EPOLL_CTL_MOD, sockfd, &ev);
} 

int clientfd_wevent_handler(struct epoll_event* event) 
{
    printf("EPOLLOUT\n");

    int sockfd = (*event).data.fd;
    write(sockfd, line, line_len);

    struct epoll_event ev;
    ev.data.fd = sockfd;
    ev.events = EPOLLIN | EPOLLET;
    epoll_ctl(epfd, EPOLL_CTL_MOD, sockfd, &ev); //TODO
}

int run_loop(int listenfd)
{
    int i, nfds;
    struct epoll_event listen_event;
    struct epoll_event events[20];

    add_listenfd_to_epoll(listenfd, &listen_event);
    listen(listenfd, LISTENQ);

    for (;;) {
        nfds = epoll_wait(epfd, events, 20, 500);
        for (i = 0; i < nfds; ++i) {
            if (events[i].data.fd == listenfd)  {
                listenfd_event_handler(listenfd);
            } else if (events[i].events & EPOLLIN) {
                clientfd_revent_handler(&events[i]);
            } else if (events[i].events & EPOLLOUT) {
                clientfd_wevent_handler(&events[i]);
            } else {
                perror("unknow event");
                exit(1);
            }
        }
    }
}

int main(int argc, char* argv[]) 
{
    char* host = get_host(argc, argv);
    int port = get_port(argc, argv);

    printf("server start: %s:%d\n", host, port);

    epfd = epoll_create(256);
    int listenfd = create_listen_socket(host, port);
    run_loop(listenfd);

    return 0;
}
