#include <stdio.h>
#include <math.h>
#include <stdlib.h>

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
    int i, sum = 0;
    for (i = 0; i < n; ++i) {
        sum += (a[i]-b[i])*(a[i]-b[i]);
    }
    return sqrt(sum);
}
void test_euc_dis()
{
    float a[] = {1,2,3,100}, b[] = {50,3,4,5};
    // np.sqrt(np.sum((np.array([1,2,3,100])-np.array([50,3,4,5]))**2))
    printf("dis = %f\n", euc_dis(4, a, b));
}

int main(int argc, char *argv[])
{
    float X[][4] = {                         // 训练集特征
        {5.9,3.0,4.2,1.5}, {6.9,3.1,5.4,2.1},
        {6.3,2.8,5.1,1.5}, {5.1,3.3,1.7,0.5},
        {5.1,3.4,1.5,0.2}, {5.7,2.5,5.0,2.0}
    };
    int y[] = {1, 2, 2, 0, 0, 2};             // 训练集标签
    int p = 3;                                // 目标分类个数
    int n = sizeof(X)/sizeof(X[0]);           // 训练数据量
    int m = sizeof(X[0])/sizeof(X[0][0]);     // 特征数量
    float test[] = {5.5,4.2,1.4,0.2};         // 测试集特征
    printf("n=%d m=%d\n", n, m);
    printf("y: ");for (int i = 0; i < n; ++i) printf("%d ", y[i]); printf("\n");
    // dists=[euc_dis(x,test) for x in X]     // 测试数据和训练集的欧式距离
    float dists[n];
    for (int i = 0; i < n; ++i) {
        dists[i] = euc_dis(m, X[i], test);
        printf("test dist: i=%d dist=%.2f y[i]=%d\n", i, dists[i], y[i]);
    }
    // idxknn= np.argsort(dists)[:k]          // 得到距离最近的 k 个训练数据索引
    int ret[n];
    argsort(n, dists, ret);
    printf("sort dists: ");for (int i = 0; i < n; ++i) printf("%d ", ret[i]); printf("\n");
    int k = 3, idxknn[k]; 
    for (int i = 0; i < k; ++i) idxknn[i] = ret[i];
    printf("idxknn: ");for (int i = 0; i < k; ++i) printf("%d ", idxknn[i]); printf("\n");
    // yknn=y[idxknn]                         // 得到对应的 k 个训练标签
    int yknn[k];
    for (int i = 0; i < k; ++i) yknn[i] = y[idxknn[i]];
    printf("yknn: ");for (int i = 0; i < k; ++i) printf("%d ", yknn[i]); printf("\n");
    // Counter(yknn).most_common(1)[0][0]     // 在 k 个标签中找出出现次数最多的标签
    int targets[p];
    for (int i = 0; i < p; ++i) targets[i] = 0;
    for (int i = 0; i < k; ++i) targets[yknn[i]] += 1;
    int max_index = -1;
    for (int i = 0; i < p; ++i) 
        if(targets[i] > max_index) max_index = targets[i];
    printf("预测的分类为：%d\n", max_index);
    return 0;
}
