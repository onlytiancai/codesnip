#include <stdio.h>
int main(int argc, char *argv[])
{
    char* s = "a(1,2,3),b(1),c,e(f(3,4),g(a,5))", *p=s, ch;
    int brackets = 0;
    printf("%s\n", s);
    while ((ch = *p++) != '\0') {
        if (ch == '(') ++brackets;
        if (ch == ')') --brackets;
        if (ch == ',' && brackets == 0) putchar('\n');
        else putchar(ch); 
    } 

    return 0;
}
