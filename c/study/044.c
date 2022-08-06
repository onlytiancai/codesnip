#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static struct sort_item {
    float data;
    int index;
};

static int cmpfunc (const void *a, const void *b)
{
   return (((struct sort_item*)a)->data - ((struct sort_item*)b)->data);
}

static int *argsort(int n, float *arr, int *ret)
{
    struct sort_item *items = (struct sort_item*)malloc(n*sizeof(struct sort_item));
    if (items == NULL) {
        perror("realloc, error");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < n; ++i) {
        items[i].data = arr[i];
        items[i].index= i;
    }
    qsort(items, n, sizeof(struct sort_item), cmpfunc);
    for (int i = 0; i < n; ++i) {
        ret[i] = items[i].index;
    }
    free(items);
}

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

static void df_load_csv(struct DataFrame *df, const char* file)
{
    FILE* fp;
    char line[255];
    char seps[] = ",\t";
    char *token;
    int arr_len = 0;
    float *arr = NULL;

    fp = fopen(file, "r");
    if (fp == NULL) {
        perror("open file error");
        exit(EXIT_FAILURE);
    }

    while(fgets(line, sizeof(line), fp)) {
        line[strlen(line) - 1] = '\0';
        arr_len = 0;
        printf("debug:getline:%s\n", line);
        token = strtok(line, seps );
        while(token != NULL) {
            arr = realloc(arr, sizeof(float) * ++arr_len);
            if (arr == NULL) {
                perror("realloc, error");
                exit(EXIT_FAILURE);
            }
            arr[arr_len-1] = atof(token);
            token = strtok(NULL, seps);
        }    
        df_add_row(df, arr_len, arr);
    }
    
    free(arr);
    fclose(fp);
}

void test_1()
{
    srand((unsigned int)time(NULL));
    struct DataFrame df;
    df_init(&df);

    df_add_row(&df, 3, (float[]){float_rand(), float_rand(), float_rand()});
    df_add_row(&df, 3, (float[]){float_rand(), float_rand(), float_rand()});
    df_add_row(&df, 3, (float[]){float_rand(), float_rand(), float_rand()});

    printf("2行2列是:%f\n", df.data[1][1]);

    df_print(&df);
    df_free(&df);

}

int main(int argc, char *argv[])
{
    struct DataFrame df;
    df_init(&df);
    df_load_csv(&df, "iris_test.csv");
    df_print(&df);
    df_free(&df);

    return 0;
}
