#include <stdio.h>
#include <unistd.h>

struct BBB { int x; int y;};
struct AAA { int a; char *s; struct BBB b;char c[6];};
void print_data(struct AAA *a) {
    printf("%s %s %d %d %d\n",
        a->s, a->c,
        a->a, a->b.x, a->b.y
    );

}

int main(int argc, char *argv[]) {
    struct AAA a = {1024, "hello", {2048, 4096}, "world"};
    print_data(&a);
    sleep(1);
    print_data(&a);
    return 0;
}
