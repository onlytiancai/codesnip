#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <stdlib.h>
char *ltrim(char *s)
{
    while(isspace(*s)) s++;
    return s;
}

char *rtrim(char *s)
{
    char* back = s + strlen(s);
    while(isspace(*--back));
    *(back+1) = '\0';
    return s;
}

char *trim(char *s)
{
    return rtrim(ltrim(s));
}

int main(int argc, char *argv[])
{
    FILE* fp;
    int buf_len = 255;
    char buffer[buf_len];
    char seps[]   = " ,\t\n";
    char *token;

    fp = fopen("045.txt", "r");

    while(fgets(buffer, buf_len, fp)) {
        printf("%s\n", trim(buffer));

        token = strtok( buffer, seps );
        while( token != NULL )
        {
            printf("%.2f\n", atof(token) );
            /* Get next token: */
            token = strtok( NULL, seps );
        }    
    }

    fclose(fp);
    return 0;
}
