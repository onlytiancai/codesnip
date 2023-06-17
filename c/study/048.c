#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>

int main( int argc, char **argv )
{
    int fd = open( __FILE__, O_RDONLY );
    if( fd == -1 ) {
        perror("open");
    }

    /*获取文件信息*/
    struct stat sb;
    if( fstat( fd, &sb ) == -1 ) {
        perror("fstat");
    }

    /*私有文件映射*/
    char *addr = mmap( NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0 );
    if( addr == MAP_FAILED ) {
        perror("mmap");
    }

    /*将addr的内容写到标准输出*/
    if( write( STDOUT_FILENO, addr, sb.st_size ) != sb.st_size ) {
        perror("write");
    }

    exit( EXIT_SUCCESS );
}
