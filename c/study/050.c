#define _GNU_SOURCE // mremap
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <time.h>
#include <string.h>

int main( int argc, char **argv )
{
    int fd = open( "050.mmap.txt", O_RDWR);
    if( fd == -1 ) {
        perror("open");
    }

    /*获取文件信息*/
    struct stat sb;
    if( fstat( fd, &sb ) == -1 ) {
        perror("fstat");
    }

    /*共享文件映射*/
    char *addr = mmap( NULL, sb.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0 );
    if( addr == MAP_FAILED ) {
        perror("mmap");
    }

    /*将addr的内容写到标准输出*/
    if( write( STDOUT_FILENO, addr, sb.st_size ) != sb.st_size ) {
        perror("write");
    }
   
    // 通过修改内存自动修改文件 
    time_t t;
    srand((unsigned) time(&t));
    size_t old_size = sb.st_size, new_size = 0;
    char buf[100];
    for (int i = 0; i < 5; ++i) {
        new_size = sprintf(buf, "%d\n", rand());
        if (new_size > old_size) {
            void *new_mapping = mremap(addr, old_size, new_size, MREMAP_MAYMOVE);
            if (new_mapping == MAP_FAILED) perror("mremap");
            addr = new_mapping;
            old_size = new_size;
        }
        ftruncate(fd, new_size);
        memset(addr, 0, new_size); 
        memcpy(addr, buf, new_size);
        if( write( STDOUT_FILENO, addr, new_size) != new_size) {
            perror("write");
        }    
        sleep(1);
    }
    exit( EXIT_SUCCESS );
}
