#include <stdio.h>
#include <unistd.h>
void print_data(char *s, int n) { printf("%s %d\n", s, n); }
int main(int argc, char *argv[]) {
        print_data("hello", 1024);
    return 0;
}
