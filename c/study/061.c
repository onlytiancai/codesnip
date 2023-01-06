#include <stdio.h>

int main(int argc, char *argv[])
{
    int i,j,c;
    i=0;
    while ((c=getchar())!=EOF) {
        if (c=='\t'){
            for (j = 0; j < (8-i%8); ++j) {
                putchar(' '); 
            }
            i = i+(8-i%8)-1;
        }
        else
            putchar(c); 
        ++i;
        if (c=='\n') i=0;
    }
    return 0;
}
