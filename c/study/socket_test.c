#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <arpa/inet.h>
#include <netinet/in.h>  //sockaddr_in 
#include <sys/socket.h>
#include <unistd.h>
#include <netdb.h>

#define RECV_BUFF_SIZE 128
#define SEND_BUF_SIZE 512

int main(int argc, const char *argv[])
{
    const char* host = argv[1];                  // 目标主机
    struct addrinfo hints;                       // 填充getaddrinfo参数
    struct addrinfo *result;                     // 存放getaddrinfo返回数据
    int r = 0;                                   // 临时存放函数返回值
    ssize_t sent_size = 0;                       // 实际发送的大小
    char send_buff[SEND_BUF_SIZE];               // 发送缓冲区
    const char *send_tpl;                        // 数据模板，%s是host占位符 
    size_t to_send_size = 0;                     // 要发送到数据大小 
    int client_fd;                               // 客户端socket
    char data_to_recv[RECV_BUFF_SIZE];           // 数据接收缓冲区
    ssize_t recv_size;                           // 已接受到的数据大小
    int i;                                       // 循环变量

    if (argc != 2) {
        printf("Usage:%s [host]\n", argv[0]);
        return 1;
    }

    send_tpl = "GET / HTTP/1.1\r\n"
               "Host: %s\r\n"
               "Accept: */*\r\n"
               "\r\n\r\n";


    memset(&hints, 0, sizeof(struct addrinfo));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = 0;
    hints.ai_protocol = 0;

    r = getaddrinfo(host, "80", &hints, &result); 
    if (r != 0) {
        printf("getaddrinfo error");
        return 3;
    }



    // 构建发送缓冲区
    if (strlen(host) + strlen(send_tpl) > SEND_BUF_SIZE - 2) { // 2 = strlen("%s")
        printf("host too long.\n");
        return 2;
    }

    to_send_size = snprintf(send_buff, SEND_BUF_SIZE, send_tpl, host);
    if (to_send_size < 0) {
        printf("snprintf error:.\n");
        return 3;
    }

    

    // 创建socket
    if ((client_fd = socket(result->ai_family, result->ai_socktype, 
                    result->ai_protocol)) == -1) {
        printf("create socket error:%d\n", client_fd);
        return -1;
    }

    printf("cerate socket ok: %d\n", client_fd);

    // 连接目标主机
    r = connect(client_fd, result->ai_addr, result->ai_addrlen);
    freeaddrinfo(result);
    if (r == -1) {
        printf("connect error.\n");
        return -2;
    }
    printf("collect ok\n");

    // 发送数据
    printf("will send:\n%s", send_buff);
    sent_size = write(client_fd, send_buff, to_send_size);
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
