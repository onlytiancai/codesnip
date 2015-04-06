#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <unistd.h>
#include <errno.h>

#include <sys/types.h>
#include <sys/socket.h>
#include <sys/select.h>  // fd_set, timeval
#include <netinet/in.h>
#include <arpa/inet.h>

#define PORT 1234
#define BACKLOG 10 
#define BUFSIZE 1024

int main(int argc, char * argv[])
{
    // ### create server socket fd
    int serverfd;

    serverfd = socket(AF_INET, SOCK_STREAM, 0);
    if (serverfd == -1)
    {
        perror("socket error"); 
        exit(EXIT_FAILURE);
    }

    // ### create server addr
    struct sockaddr_in server_addr;
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    server_addr.sin_port = htons(PORT);
    memset(server_addr.sin_zero, '\0', sizeof(server_addr.sin_zero));


    // ### bind server socket fd 
    if (bind(serverfd, (struct sockaddr*)(&server_addr), sizeof(server_addr)) == -1)
    {
        perror("bind error");
        exit(EXIT_FAILURE);
    }

    // ### listen server socket fd
    if (listen(serverfd, BACKLOG) == -1)
    {
        perror("listen error");
        exit(EXIT_FAILURE);
    }
    printf("listen port:%d \n", PORT);

    // ### select
    fd_set fdsets;
    int maxsock;                    
    struct timeval tv;              
    struct sockaddr_in client_addr;
    int fdarr[BACKLOG];
    char buf[BUFSIZE];

    socklen_t sin_size = sizeof(client_addr);
    maxsock = serverfd;
    int conn_amount = 0; 
    int i;
    int ret;
    int clientfd;

    // 清零fdarr
    for (i = 0; i < BACKLOG; i ++)
    {
        fdarr[i] = 0;
    }

    while(1){
        FD_ZERO(&fdsets);
        FD_SET(serverfd, &fdsets);

        tv.tv_sec = 30;
        tv.tv_usec = 0;

        // 把client socket fd监控起来
        for( i = 0; i < BACKLOG; i ++) 
        {
            if (fdarr[i] != 0){
                FD_SET(fdarr[i], &fdsets); 
            }
        } 

        // 查看select结果
        ret = select(maxsock + 1, &fdsets, NULL, NULL, &tv);
        if (ret < 0)
        {
            perror("select error"); 
            exit(EXIT_FAILURE);
        }else if (ret == 0)
        {
            printf("select timeout \n");
            continue;
        }

        // 接受客户端发送过来的数据
        for (i = 0; i < conn_amount; i++)
        {
            if (FD_ISSET(fdarr[i], &fdsets))
            {
                ret = read(fdarr[i], buf, BUFSIZE);
                if (ret <= 0)
                {
                    printf("client %d close.", i);
                    close(fdarr[i]);
                    FD_CLR(fdarr[i], &fdsets);
                    fdarr[i] = 0;
                }else {
                    buf[ret] = '\0';
                    printf("recv data[%d]:%s \n", i, buf);
                }
            }
        }

        // 检查新连接
        if (FD_SET(serverfd, &fdsets))
        {
            clientfd = accept(serverfd, (struct sockaddr *)&client_addr, &sin_size);
            if (clientfd < 0)
            {
                perror("accept error"); 
                continue;
            }
            
            // 添加到监控队列里
            if (conn_amount < BACKLOG){
                FD_SET(clientfd, &fdsets);
                fdarr[conn_amount++] = clientfd;
                printf("accept conn[%d]:%s %d \n", conn_amount, 
                        inet_ntoa(client_addr.sin_addr),
                        ntohs(client_addr.sin_port));

                if (clientfd > maxsock)
                {
                    maxsock = clientfd;
                }
            } else{  // 过载保护
                printf("max conn, exit\n");
                write(clientfd, "bye", 4);
                close(clientfd);
                break; // TODO: 这里为什么奥break呢
            }
        }
    }

    // 关闭所有客户端连接
    for (i = 0; i < BACKLOG; i++){
        if (fdarr[i] != 0) {
            close(fdarr[i]); 
        }
    }


    return 0;
}
