#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
int main(void) {
    FILE * fp = fopen("cat2.c","r");
    if (fp == NULL){
        perror("fopen");
        return 1; 
    }
 
    struct stat stats;
    if (fstat(fileno(fp),&stats)== -1){ // POSIX only 
        perror("fstat"); 
        return 1;
    } 

    printf("BUFSIZ 是%d,但最佳块大小是%ld\n", BUFSIZ, stats.st_blksize);
    if (setvbuf(fp, NULL, _IOFBF, stats.st_blksize) != 0){
        perror("setvbuf failed");
        return 1; 
    }
 
    int ch;
    while ((ch = fgetc(fp)) != EOF) {
        putchar(ch);
    };
    //读取整个文件：使用truss / strace //观察read（2）使用的系统调用 
    fclose(fp);
}
