#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <arpa/inet.h>
#include <netinet/in.h>  //sockaddr_in 
#include <sys/socket.h>
#include <unistd.h>

#define RECV_BUFF_SIZE 128 

int main()
{
    int client_fd;                                     // 客户端socket
    struct sockaddr_in s_addr;                         // 目标主机地址
    char * data_to_send = "GET / HTTP/1.1\r\n"         // 要发送到数据
                          "Host: www.baidu.com\r\n"
                          "Accept: */*\r\n"
                          "\r\n\r\n";
    size_t to_send_size = strlen(data_to_send);  // 要发送数据的大小
    ssize_t sent_size = 0;                        // 实际发送的大小

    char data_to_recv[RECV_BUFF_SIZE];                 // 数据接收缓冲区
    ssize_t recv_size;                                     // 已接受到的数据大小

    int i;                                             // 循环变量

    // 创建socket
    if ((client_fd = socket(AF_INET, SOCK_STREAM, 0)) == -1) {
        printf("create socket error:%d\n", client_fd);
        return -1;
    }

    printf("cerate socket ok: %d\n", client_fd);

    // 构建目标主机地址
    memset(&s_addr, 0, sizeof(struct sockaddr_in));
    s_addr.sin_family = AF_INET;
    s_addr.sin_addr.s_addr = inet_addr("180.97.33.71"); 
    s_addr.sin_port = htons(80);
    printf("will collect s_addr=%#x, port=%#x\n", (unsigned int)s_addr.sin_addr.s_addr, 
           (unsigned int)s_addr.sin_port);

    // 连接目标主机
    if (connect(client_fd, (struct sockaddr *)(&s_addr), (socklen_t)sizeof(struct sockaddr)) == -1) {
        printf("connect error.\n");
        return -2;
    }
    printf("collect ok\n");

    // 发送数据
    printf("will send:\n%s", data_to_send);
    sent_size = write(client_fd, data_to_send, (size_t)strlen(data_to_send));
    if (sent_size < 0) {
        printf("send data error.\n");
        return -3;
    }else if((size_t)sent_size != to_send_size){
         printf("not all send.\n");
    }
    printf("send data ok.\n");

    // 接收数据
    recv_size = read(client_fd, &data_to_recv, RECV_BUFF_SIZE);
    if (recv_size < 0) {
        printf("recv data error:%d\n", (int)recv_size);
        return -4;
    }
    if (recv_size == 0) {
        printf("recv 0 size data.\n");
        return -5;
    }
    for (i = 0; i < RECV_BUFF_SIZE - 1; i++) {
        if (data_to_recv[i] == '\r' && data_to_recv[i+1] == '\n') {
            data_to_recv[i] = '\0';
        }
    }
    printf("recv data:%s\n", data_to_recv);

    // 关闭socket
    if(close(client_fd) < 0){
         printf("close socket errors\n");
         return -6;
    }
    printf("close socket ok\n");
    
    return 0;
}
