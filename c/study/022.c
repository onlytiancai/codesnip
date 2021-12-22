#include <stdio.h>

int main(int argc, char *argv[])
{
    FILE *fp, *fp_out;
    int c, count = 0;

    fp = fopen("number.in", "r");
    for (int c = fgetc(fp); !feof(fp); c = fgetc(fp)) {
        if (c == '1') count++;
    }
    fclose(fp);
    printf("%d\n", count);

    fp_out = fopen("number.out", "w");
    fprintf(fp_out, "%d\n", count);
    fclose(fp_out);

    return 0;
}
