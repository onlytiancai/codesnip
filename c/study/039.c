#include <stdio.h>

int main(int argc, char *argv[])
{
    int a, b, l, r, m, x, i = 0, f=0;
    scanf("%d %d", &a, &b);
    printf("%d %d\n", a, b);

    l = 1, r = 999;
    while (l < r) {
        m = l + (r-l) / 2;
        printf("l=%d m=%d r=%d x=%d\n", l, m, r, x);

        x = b/100*m*m + b/10%10*m + b%100%10;
        if (x > a) {
            r = m-1;
        } else if (x < a) {
            l = m+1;
        } else {
            f = m;
            break;
        }

        if (++i > 20) {
            printf("over flow\n");
            break;
        }
    }

    if (f != 0) {
        printf("%d\n", f);
    } else {
        printf("not found\n");
    }
    
    return 0;
}
