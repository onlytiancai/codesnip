#include <stdio.h>
#include <strings.h> // bzero
#include <arpa/inet.h> // struct sockaddr_in
#include <netinet/in.h> // htons 

int main() {
    struct sockaddr_in server_addr;

    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(8001);
    server_addr.sin_addr.s_addr = INADDR_ANY;
    bzero(&(server_addr.sin_zero), 8);

    int struct_len = sizeof(struct sockaddr_in);

    int fd = socket(AF_INET, SOCK_STREAM, 0);


    return 0;
}
