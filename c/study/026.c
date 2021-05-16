#include <stdio.h>

int main()
{
    char *arr[] = {"aaa"}; 
    char *s = arr[0];
    char **p = &arr[0];
    arr[0] = "bbb";
    printf("%s %s %s\n", arr[0], s, *p);
    return 0;
}
