#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <string.h>



int main( int argc, char **argv )
{
    /*获取虚拟设备的文件描述符*/
    int fd = open( "/dev/zero", O_RDWR );
    if( fd == -1 ) {
        perror("open");
    }

    int *addr = mmap( NULL, sizeof(int), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0 );
    if( addr == MAP_FAILED ) {
        perror("mmap");
    }

    if( close( fd ) == -1 ) {
        perror("close");
    }

    *addr = 1;

    switch( fork() ) {
        case -1:
            perror("fork");
            break;
        case 0:
            printf("child *addr = %d\n", *addr);
            (*addr)++;

            /*解除映射*/
            if( munmap(addr, sizeof(int)) == -1 ) {
                perror("munmap");
            }
            _exit( EXIT_SUCCESS );
            break;
        default:
            /*等待子进程结束*/
            if( wait(NULL) == -1 ) {
                perror("wait");
            }

            printf("parent *addr = %d\n", *addr );
            if( munmap( addr, sizeof(int) ) == -1 ) {
                perror("munmap");
            }
            exit( EXIT_SUCCESS );
            break;
    }
}
