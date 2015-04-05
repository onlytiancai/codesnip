#include <stdio.h>

void print_binary (char c) {
    int i;
    for (i = 7; i >= 0; i--) {
        printf("%d", (c >> i) & 1);
    }
    printf("\n");
}

int main(void)
{
    char c = 0;

    printf("0 = ");
    print_binary(c);

    c = c | 1 << 7;
    printf("c | 1 << 7: ");
    print_binary(c);

    c = c | 3 << 5;
    print_binary(3);
    printf("c | 3 << 5: ");
    print_binary(c);

    return 0;
}
