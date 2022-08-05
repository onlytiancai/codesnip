#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct DataFrame{
    int N;
    int M;
    int _max_row;
    double **data;
}; 

static void df_init(struct DataFrame *df)
{
    df->M = 0;
    df->N = 0;
    df->_max_row = 0;
}

static void df_free(struct DataFrame *df)
{
    int i;
    for (i = 0; i < df->N; ++i) {
        free(df->data[i]); 
    }
    free(df->data);
}

static void df_auto_extend(struct DataFrame *df)
{
    double **new_data;
    int new_max_row;
    if (df->N == df->_max_row) {
        printf("debug:df_auto_extend: %d %d\n", df->N, df->_max_row);
        new_max_row = df->_max_row == 0 ? 1 : df->_max_row * 2;
        new_data = (double**)malloc(new_max_row * sizeof(double**));
        memcpy(new_data, df->data, df->_max_row * sizeof(double*));
        if (df->_max_row >0) free(df->data);
        df->data = new_data;
        df->_max_row = new_max_row;
    }
}

static void df_add_row(struct DataFrame *df, int m, double *row)
{
    if (df->M != 0 && m != df->M) {
       fprintf(stderr, "All rows should have equal columns:%d %d\n", df->M, m);
       exit(EXIT_FAILURE);
    }
    df_auto_extend(df);
    int bytes = m * sizeof(double);
    double* new_row = (double*)malloc(bytes); 
    memcpy(new_row, row, bytes);
    df->M = m;
    df->data[df->N++] = new_row;
}

int main(int argc, char *argv[])
{
    struct DataFrame df;
    df_init(&df);

    double row[] = {1, 2, 3};
    df_add_row(&df, 3, row);
    df_add_row(&df, 3, row);
    df_add_row(&df, 3, row);

    printf("%s %d %d %d\n", "df", df.M, df.N, df._max_row);
    df_free(&df);
    return 0;
}
