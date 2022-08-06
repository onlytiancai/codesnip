#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

struct sort_item {
    float data;
    int index;
};

int cmpfunc (const void *a, const void *b)
{
    float x =  ((struct sort_item*)a)->data;
    float y =  ((struct sort_item*)b)->data;
    if (x>y) return 1;
    if (x<y) return -1;
    return 0;
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

// np.sqrt(np.sum((instance1-instance2)**2))
float euc_dis(int n, float *a, float *b)
{
    float sum = 0;
    for (int i = 0; i < n; ++i) {
        sum += (a[i]-b[i])*(a[i]-b[i]);
        //printf("debug:euc dis:%i:sum=%f, a[i]=%f, b[i]=%f\n", i, sum, a[i], b[i]);
    }
    return sqrt(sum);
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
        //printf("debug:df_auto_extend: %d %d\n", df->N, df->_max_row);
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
        //printf("debug:getline:%s\n", line);
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

// X:训练集，前面的列是特征，最后一列是标签，
// test:单个测试数据特征，k: knn 的 k, p: 分类数 
int knn_classify(struct DataFrame *X, float *test, int k, int p)
{
    int n = X->N;           // 训练数据量
    int m = X->M -1;           // 特征数量

    // 填充训练集标签
    int *y = (int*)malloc(n*sizeof(int));
    if (y== NULL) {
        perror("malloc, error");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < n; ++i) y[i] = X->data[i][m];

    //printf("n=%d m=%d\n", n, m);
    //printf("y: ");for (int i = 0; i < n; ++i) printf("%d ", y[i]); printf("\n");
    //printf("test: ");for (int i = 0; i < m; ++i) printf("%f ", test[i]); printf("\n");
    // dists=[euc_dis(x,test) for x in X]     // 测试数据和训练集的欧式距离
    float dists[n];
    for (int i = 0; i < n; ++i) {
        dists[i] = euc_dis(m, X->data[i], test);
        //printf("test dist: i=%d dist=%.2f y[i]=%d\n", i, dists[i], y[i]);
    }
    // idxknn= np.argsort(dists)[:k]          // 得到距离最近的 k 个训练数据索引
    int ret[n];
    argsort(n, dists, ret);
    //printf("sort dists: ");for (int i = 0; i < n; ++i) printf("%d ", ret[i]); printf("\n");
    int idxknn[k]; 
    for (int i = 0; i < k; ++i) idxknn[i] = ret[i];
    //printf("idxknn: ");for (int i = 0; i < k; ++i) printf("%d ", idxknn[i]); printf("\n");
    // yknn=y[idxknn]                         // 得到对应的 k 个训练标签
    int yknn[k];
    for (int i = 0; i < k; ++i) yknn[i] = y[idxknn[i]];
    //printf("yknn: ");for (int i = 0; i < k; ++i) printf("%d ", yknn[i]); printf("\n");
    // Counter(yknn).most_common(1)[0][0]     // 在 k 个标签中找出出现次数最多的标签
    int targets[p];
    for (int i = 0; i < p; ++i) targets[i] = 0;
    for (int i = 0; i < k; ++i) targets[yknn[i]] += 1;
    int max_count = -1, max_index;
    for (int i = 0; i < p; ++i) {
        if(targets[i] > max_count) {
            max_count= targets[i];
            max_index = i;
        }
    }
    free(y);
    return max_index;
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
    struct DataFrame df_train;
    df_init(&df_train);
    df_load_csv(&df_train, "iris_training.csv");
    df_print(&df_train);

    struct DataFrame df_test;
    df_init(&df_test);
    df_load_csv(&df_test, "iris_test.csv");

    int ret;
    int k = 10;                       // 距离最近的 k 个数据，KNN 中的 k
    int p = 3;                        // 目标分类的个数
    float test_features[df_test.M-1]; // 测试特征
    int test_target;                  // 测试标签
    int total_count = 0, ok_count = 0;
    for (int i = 0; i < df_test.N; ++i) {
        for (int j = 0;  j < df_test.M - 1; j++) {
            test_features[j] = df_test.data[i][j];
        }

        test_target = df_test.data[i][df_test.M-1];
        int ret = knn_classify(&df_train, test_features, k, p);
        printf("预测分类：%d, 实际分类：%d\n", ret, test_target);

        total_count += 1;
        if (ret == test_target) ok_count += 1;
    }
    printf("测试%d个数据，正确%d个，准确率：%.2f%%\n", 
            total_count, ok_count, (float)ok_count/total_count*100);

    df_free(&df_train);
    df_free(&df_test);
    return 0;
}
