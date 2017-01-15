/**
 * epoll 学习
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

static int epfd;
static char line[MAXLINE];
static int line_len;


char* get_host(int argc, char* argv[])
{
    if (argc < 3) return "0.0.0.0";
    return argv[1];
}

int get_port(int argc, char* argv[])
{
    if (argc < 2) return DEFAULT_PORT;
    if (argc == 2) return (int) strtol(argv[1], (char **)NULL, 10);
    if (argc > 2) return (int) strtol(argv[2], (char **)NULL, 10);
    
}

int add_listenfd_to_epoll(int listenfd)
{
    struct epoll_event ev;
    ev.data.fd = listenfd;
    ev.events = EPOLLIN | EPOLLET;
    ev.events = EPOLLIN;
    epoll_ctl(epfd, EPOLL_CTL_ADD, listenfd, &ev);
}

void setnonblocking(int sock)
{
    int opts;
    opts=fcntl(sock,F_GETFL);
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

int create_listen_socket(char* host, int port)
{
    int listenfd = socket(AF_INET, SOCK_STREAM, 0);
    setnonblocking(listenfd);
    add_listenfd_to_epoll(listenfd);

    struct sockaddr_in serveraddr;
    bzero(&serveraddr, sizeof(serveraddr));
    serveraddr.sin_family = AF_INET;

    inet_aton(host, &(serveraddr.sin_addr));
    serveraddr.sin_port = htons(port);

    bind(listenfd,(struct sockaddr *)&serveraddr, sizeof(serveraddr));
    listen(listenfd, LISTENQ);

    return listenfd;
}

int listenfd_event_handler(int listenfd)
{
    socklen_t clilen;
    struct sockaddr_in clientaddr;

    int connfd = accept(listenfd, (struct sockaddr *)&clientaddr, &clilen);
    if (connfd < 0) {
        perror("connfd < 0");
        exit(1);
    }

    setnonblocking(connfd);

    char *str = inet_ntoa(clientaddr.sin_addr);
    printf("accapt a connection from %s \n", str);

    struct epoll_event ev;
    ev.data.fd = connfd;
    ev.events = EPOLLIN | EPOLLET;
    epoll_ctl(epfd, EPOLL_CTL_ADD, connfd, &ev);
}

int clientfd_revent_handler(struct epoll_event* event)
{
    printf("EPOLLIN\n");
    int sockfd;
    struct epoll_event ev = *event;

    if ((sockfd = ev.data.fd) < 0) 
        return;

    if ((line_len = read(sockfd, line, MAXLINE)) < 0) {
        if (errno == ECONNRESET) {
            close(sockfd);
            ev.data.fd = -1;
        } else {
            printf("readline error\n");
        }
    } else if (line_len == 0) {
        printf("client closed\n");
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
    struct epoll_event events[20];

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
