#include <stdio.h>
#include <time.h>

#define N 6
int main(int argc, char* argv[]) {
    srand(time(NULL));
    printf("hello 012\n");
    
    int data[] = {1,3,5,7,9,11,13,15,17,19,21};
    int len = sizeof(data) / sizeof(int);
    int choice[N];
    int i, j, index, hit, d;
    for (i = 0; i < N; i++) {
        while(1) {
            hit = 0;
            index = rand() % len;
            d = data[index];
            printf("debug: i=%d index=%d d=%d\n", i, index, d);
            for (j = 0; j < i; j++) {
                if (choice[j] == d) {
                    printf("debug: exists:%d \n", d);
                    hit = 1;
                    break;
                }
            }
            if (hit==0) break;
        }
        choice[i] = d;
    }

    printf("result: ");
    for (i = 0; i < N; i++) {
        printf("%d ", choice[i]);
    }
    printf("\n");

    return 0;
}
