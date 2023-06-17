#include <stdio.h>

void scoped(int * pvariable) {
    printf("variable (%d) goes out of scope\n", *pvariable);
}

int main(void) {
    printf("before scope\n");
    {
        int watched __attribute__((cleanup (scoped)));
        watched = 42;
    }
    printf("after scope\n");
}
