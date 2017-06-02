#include <stdio.h>

char str[20];

int main()
{
    FILE *f;
    char ch;
    size_t ret_code;

    f = fopen("file.txt", "w");
    for (ch = '0'; ch <= '9'; ch++) {
        fputc(ch, f);
    }
    fclose(f);

    f = fopen("file.txt", "r");
    ret_code = fread(str, 1, 10, f);
    printf("fread return: %d \n", ret_code);
    puts(str);

    rewind(f);
    fread(str, 1, 10, f);
    puts(str);

    fclose(f);
    return 0;
}
