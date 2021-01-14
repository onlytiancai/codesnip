#include <stdio.h>

int main(int argc, char *argv[])
{
    FILE *fp = NULL;
    fprintf(fp, "%s\n", "hello"); 
    fclose(fp);
    return 0;
}
