#include <stdlib.h>     // exit, malloc, free, rand
#include <stdio.h>      // stdin, stdout, stderr, printf, getchar
#include <errno.h>      // errno, EAGAIN, EBADF, ECONNABORTED, ECONNREFUSED, ECONNRESET
#include <string.h>     // strcpy, strlen, strcmp, memcpy, memset, strerror, bzero

#include <unistd.h>     // fork, fcntl, getpid, exec, read, write, close
#include <netdb.h>      // gethostbyname, gethostbyaddr, getaddrinfo

#include <sys/socket.h> // socket, bind, accept, connect, send, recv, shutdown
#include <netinet/in.h> // in_port_t, in_addr_t, sockaddr_in, INADDR_ANY
#include <sys/types.h>  // size_t, timer_t, pthread_t
#include <arpa/inet.h>  // htonl, htons, ntohl, ntohs, inet_addr, inet_ntoa, inet_aton

int main(int argc, const char *argv[])
{
    int serverfd, clientfd;
    struct sockaddr_in server_addr;
    struct sockaddr_in client_addr;
    int sin_size, portnumber;

    char hello[] = "Hello, Are you Fine?\n";

    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s portnumber\n", argv[0]);
        exit(1);
    }

    if ((portnumber = atoi(argv[1])) < 0) 
    {
        fprintf(stderr, "Usage: %s portnumber\n", argv[0]);
        exit(1);
    }

    if ((serverfd = socket(AF_INET, SOCK_STREAM, 0)) == -1) // protocol = 0
    {
        fprintf(stderr, "Socket error:%s\n", strerror(errno));
        exit(1);
    
    }

    memset(&server_addr, 0, sizeof(struct sockaddr_in));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    server_addr.sin_port = htons(portnumber);


    if (bind(serverfd, (struct sockaddr *)(&server_addr), sizeof(struct sockaddr)) == -1)
    {
        fprintf(stderr, "Bind Error:%s \n", strerror(errno));
        exit(1);
    }

    fprintf(stdout, "Server bind: %s:%d \n", inet_ntoa(server_addr.sin_addr), portnumber);

    if (listen(serverfd, 5) == -1) // baklog = 5
    {
        fprintf(stderr, "Listen error: %s \n", strerror(errno));
        exit(1);
    }

    while(1)
    {
        sin_size = sizeof(struct sockaddr_in);
        if ((clientfd = accept(serverfd, (struct sockaddr *)(&client_addr), &sin_size)) == -1)
        {
            fprintf(stderr, "Accept Error:%s \n", strerror(errno));
            exit(1);
        }

        fprintf(stdout, "Server Get Connection From %s\n", inet_ntoa(client_addr.sin_addr));

        if (write(clientfd, hello, strlen(hello)) == -1) 
        {
            fprintf(stderr, "Write Error:%s \n", strerror(errno));
            exit(1);
        }

        close(clientfd);
    }

    close(serverfd);
    return 0;
}
