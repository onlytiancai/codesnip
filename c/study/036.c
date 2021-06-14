#include <stdio.h>
#include <string.h>

const int MAX_LINE = 50; 
int main()
{
    char line[MAX_LINE];
    while (1) {
        fgets(line, MAX_LINE, stdin);
        line[strcspn(line, "\n")] = 0;
        if (strcmp(line, "quit") == 0) {
            break;
        }
        printf("%s\n", line);
    }
    return 0;
}
