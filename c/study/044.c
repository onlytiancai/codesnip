#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

float float_rand()
{
    float min = 0, max = 10;
    float scale = rand() / (float) RAND_MAX; /* [0, 1.0] */
    return min + scale * ( max - min );      /* [min, max] */
}

struct DataFrame{
    int N;
    int M;
    int _max_row;
    float **data;
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

static void df_print(struct DataFrame *df)
{
    printf("print df: M=%d N=%d _max_row=%d\n", df->M, df->N, df->_max_row);
    int i,j;
    for (i = 0; i < df->N; ++i) {
        for (j = 0; j < df->M; ++j) {
            printf("%.2f\t", df->data[i][j]);
        }
        printf("\n");
    }
}

static void df_auto_extend(struct DataFrame *df)
{
    float **new_data;
    int new_max_row;
    if (df->N == df->_max_row) {
        printf("debug:df_auto_extend: %d %d\n", df->N, df->_max_row);
        new_max_row = df->_max_row == 0 ? 1 : df->_max_row * 2;
        new_data = (float**)malloc(new_max_row * sizeof(float**));
        memcpy(new_data, df->data, df->_max_row * sizeof(float*));
        if (df->_max_row >0) free(df->data);
        df->data = new_data;
        df->_max_row = new_max_row;
    }
}

static void df_add_row(struct DataFrame *df, int m, float*row)
{
    if (df->M != 0 && m != df->M) {
       fprintf(stderr, "All rows should have equal columns:%d %d\n", df->M, m);
       exit(EXIT_FAILURE);
    }
    df_auto_extend(df);
    int bytes = m * sizeof(float);
    float* new_row = (float*)malloc(bytes); 
    memcpy(new_row, row, bytes);
    df->M = m;
    df->data[df->N++] = new_row;
}

int main(int argc, char *argv[])
{
    srand((unsigned int)time(NULL));
    struct DataFrame df;
    df_init(&df);

    df_add_row(&df, 3, (float[]){float_rand(), float_rand(), float_rand()});
    df_add_row(&df, 3, (float[]){float_rand(), float_rand(), float_rand()});
    df_add_row(&df, 3, (float[]){float_rand(), float_rand(), float_rand()});

    df_print(&df);
    df_free(&df);
    return 0;
}
