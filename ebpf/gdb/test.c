#include <stdio.h>

struct BBB { int x; int y;};
typedef struct BBB BBB_t;
struct AAA { int a;char *s; BBB_t b; char c[6];};

void print_data(char *s, struct AAA *a) { 
    printf("%s %s %d %d\n", s, a->s, a->b.x,a->b.y);
}

int main(int argc, char *argv[]) {
    char *s = "hello";
    struct AAA a = {.a=888, .s=s, {.x=1024, .y=2048}, .c="world"};
    print_data(s, &a);
    return 0;
}
