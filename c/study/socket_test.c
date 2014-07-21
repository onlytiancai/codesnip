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

struct addrinfo* get_addr(const char *host, const char *port){
    struct addrinfo hints;     // 填充getaddrinfo参数
    struct addrinfo *result;   // 存放getaddrinfo返回数据

    memset(&hints, 0, sizeof(struct addrinfo));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = 0;
    hints.ai_protocol = 0;

    if(getaddrinfo(host, port, &hints, &result) != 0) {
        printf("getaddrinfo error");
        exit(1);
    }
    return result;
}

int create_socket(const struct addrinfo * result) {
    int fd;

    if ((fd = socket(result->ai_family, result->ai_socktype, result->ai_protocol)) == -1) {
        printf("create socket error:%d\n", fd);
        exit(-1);
    }
    printf("cerate socket ok: %d\n", fd);
    return fd;
}

int get_send_data(char * buf, size_t buf_size, const char* host) {
    const char *send_tpl;                        // 数据模板，%s是host占位符 
    size_t to_send_size;                         // 要发送到数据大小 

    send_tpl = "GET / HTTP/1.1\r\n"
               "Host: %s\r\n"
               "Accept: */*\r\n"
               "\r\n\r\n";

    // 格式化后的长度必须小于buf的大小，因为snprintf会在最后填个'\0'
    if (strlen(host) + strlen(send_tpl) - 2 >= buf_size) { // 2 = strlen("%s")
        printf("host too long.\n");
        exit(-1);
    }

    to_send_size = snprintf(buf, buf_size, send_tpl, host);
    if (to_send_size < 0) {
        printf("snprintf error:%s.\n", to_send_size);
        exit(-2);
    }

    return to_send_size;
}

int connect_host(int fd, const struct addrinfo* addr) {
    if (connect(fd , addr->ai_addr, addr->ai_addrlen) == -1) {
        printf("connect error.\n");
        exit(-1);
    }
    printf("collect ok\n");
    return 0;
}

int send_data(int fd, const char *data, size_t size) {
    size_t sent_size;
    printf("will send:\n%s", data);
    sent_size = write(fd, data, size);
    if (sent_size < 0) {
        printf("send data error.\n");
        exit(-1);
    }else if(sent_size != size){
         printf("not all send.\n");
         exit(-2);
    }
    printf("send data ok.\n");
    return sent_size;
}

int recv_data(int fd, char* buf, int size) {
    int i;
    int recv_size = read(fd, buf, size);
    if (recv_size < 0) {
        printf("recv data error:%d\n", (int)recv_size);
        exit(-1);
    }
    if (recv_size == 0) {
        printf("recv 0 size data.\n");
        exit(-2);
    }
    // 只取HTTP first line
    for (i = 0; i < size - 1; i++) {
        if (buf[i] == '\r' && buf[i+1] == '\n') {
            buf[i] = '\0';
        }
    }
    printf("recv data:%s\n", buf);
}

int close_socket(int fd) {
    if(close(fd) < 0){
         printf("close socket errors\n");
         exit(-1);
    }
    printf("close socket ok\n");
}

int main(int argc, const char *argv[])
{
    const char* host = argv[1];                  // 目标主机
    char send_buff[SEND_BUF_SIZE];               // 发送缓冲区
    char recv_buf[RECV_BUFF_SIZE];               // 接收缓冲区
    size_t to_send_size = 0;                     // 要发送数据大小 
    int client_fd;                               // 客户端socket
    struct addrinfo *addr;                       // 存放getaddrinfo返回数据

    if (argc != 2) {
        printf("Usage:%s [host]\n", argv[0]);
        return 1;
    }


    addr = get_addr(host, "80");
    client_fd = create_socket(addr);
    connect_host(client_fd, addr);
    freeaddrinfo(addr);

    to_send_size = get_send_data(send_buff, SEND_BUF_SIZE, host);
    send_data(client_fd, send_buff, to_send_size);

    recv_data(client_fd, recv_buf, RECV_BUFF_SIZE);

    close(client_fd);
    return 0;
}


