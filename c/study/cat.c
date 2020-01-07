#include <stdio.h>
#include <string.h>

static char* program_name;
static int usage() {
    printf("%s filename\n", program_name);
    return 1;
}

int main(int argc, char **argv) {
    // 打印参数
    int i;
    char **p = argv;
    // 打开文件
    FILE *fp = NULL;
    // 读取文件内容
    size_t buff_size = 20;
    char buff[buff_size + 1];
    size_t n ;
    // 设置程序名称
    program_name = *argv;

    // 参数检查
    if (argc < 2) return usage();

    for (i = 0; i < argc; i++) {
        printf("arg:%d=%s\n", i, *p++); 
    }

    fp = fopen(argv[1], "r");
    if (fp == NULL) {
        perror("open file error");
        return 1; 
    }

    while (feof(fp) != 0) {
        memset(buff, 0, sizeof(buff));
        n = fread(buff, sizeof(char), buff_size, fp);

        // 处理读取错误
        if (n != buff_size && feof(fp) != 0) {
            perror("read file error");
            (void)fclose(fp);
            return 1;
        }

        // 打印读取出来的字符
        if (n > 0) {
            printf("%s", buff);
        }
    }

    // 关闭文件
    (void)fclose(fp);
    return 0;
}
