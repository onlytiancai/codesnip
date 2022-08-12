#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[])
{
    FILE *fp;
    char line[128]; 

    fp = fopen("test_data.csv", "r");
    if(fp == NULL) {
        perror("fopen error");
        return(-1);
    }

    char *s = NULL;
    int i = 0;
    char *str, *token;
    while (fgets(line, 128, fp) !=NULL && ++i < 10) {
        s = strchr(line, '\n');
        if (s) {
            *s = '\0';
        }
        
        printf("location:%ld\n", ftell(fp)-strlen(line)-1);
        puts(line);
        str = line;
        token = strtok(str, ",");
        if (token != NULL) printf("num=%d\n", atoi(token));

        token = strtok(NULL, ",");
        if (token != NULL) printf("data=%s\n", token);
    }

    fclose(fp);

    return 0;
}
