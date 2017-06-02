// echo -en "aa bb cc\n11 22 33\n" | ./a.out
#include <stdio.h>
#include <stdlib.h>

int main(void)
{
    int line_count = 0;
    int word_count = 0;
    int char_count = 0;
    char c, last_c;

    while ((c = getchar()) != EOF) {
        last_c = c;
        char_count++;
        if (isspace(c))
        {
            printf("%d\n", c);
            word_count ++;
            if (c == '\n') {
                line_count ++;
            }
        }
    }

    if (last_c != '\n') line_count++;

    printf("line_count=%d, word_count=%d, char_count=%d\n", line_count, word_count, char_count);

    return 0;
}
