#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arpa/inet.h>
#include <sys/socket.h>

#define BUF_SIZE 12 

void bin_print(char c) {
    int i;
    for (i = 7; i >= 0; i--) {
        printf("%d", (c >> i) & 1);
    }
    printf("\n");
}


void set_send_buf(char* buf, int size, const char* domain) {
    char *p = buf;

    uint16_t id = 1024;
    uint16_t flag = 0;
    uint16_t qdcount = 1;
    uint16_t ancount = 0;
    uint16_t nscount = 0;
    uint16_t arcount = 0;

    int qr = 0;         // 1 bit, 0表查询，1表应答
    int opcode = 0;     // 4 bit, 查询类型，0表标准查询，1表反向查询
    int aa = 0;         // 1 bit, 是否权威应答
    int tc = 0;         // 1 bit, 是否截断超出的部分
    int rd = 1;         // 1 bit, 查询时是否启用递归查询，
    int ra = 0;         // 1 bit, 应答时表示服务器是否支持递归查询 
    int z = 0;          // 3 bit, 保留位，必须为0
    int rccode = 0;     // 4 bit, 应答码，0:成功，1:格式错，2:DNS错，
                        //                3:域名不存在，4:DNS不支持该类查询，
                        //                5:DNS剞劂查询

    flag = flag | (qr << 15);
    flag = flag | (opcode << 11);
    flag = flag | (aa << 10);
    flag = flag | (tc << 9);
    flag = flag | (rd << 8);
    flag = flag | (ra << 7);
    flag = flag | (z << 4);
    flag = flag | (rccode << 0);

    memcpy(p, &id, sizeof(id));
    p += 2;
    memcpy(p, &flag, sizeof(flag));
    p += 2;
    memcpy(p, &qdcount, sizeof(qdcount));
    p += 2;
    memcpy(p, &ancount, sizeof(ancount));
    p += 2;
    memcpy(p, &nscount, sizeof(nscount));
    p += 2;
    memcpy(p, &arcount, sizeof(arcount));
    p += 2;

}

int main(int argc, const char *argv[])
{
    int fd, i;
    ssize_t r;
    char buf[BUF_SIZE];
    struct sockaddr_in addr;
    const char* host = "114.114.114.114";

    fd = socket(AF_INET, SOCK_DGRAM, 0); 

    addr.sin_family = AF_INET;
    addr.sin_port = htons(53);
    addr.sin_addr.s_addr = inet_addr(host);

    memset(buf, 0, BUF_SIZE);
    set_send_buf(buf, BUF_SIZE, "www.baidu.com");
    for (i = 0; i < BUF_SIZE; i++) {
        bin_print(buf[i]);
    }
    r = sendto(fd, buf, BUF_SIZE, 0, (struct sockaddr *)&addr, sizeof(struct sockaddr_in));

    return 0;
}
