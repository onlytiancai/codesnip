/*
 * This time, you are supposed to find A+B where A and B are two matrices,
 * and then count the number of zero rows and columns.
 *
 * 输入
 * The input consists of several test cases, 
 * each starts with a pair of positive integers M and N (≤10) which are the number
 * of rows and columns of the matrices, respectively. Then 2*M lines follow, ea
 *
 *
 * 输出
 * For each test case you should output in one line the total number of zero rows and columns of A+B.
 *
 * ## input
 * 2 2
 * 1 1
 * 1 1
 * -1 -1
 * 10 9
 * 2 3
 * 1 2 3
 * 4 5 6
 * -1 -2 -3
 * -4 -5 -6
 * 0
 * ## output
 * 1
 * 5
 * */

#include <stdio.h>
#include <stdlib.h>

// 从 stdin 读入 n 个 int 到 arr
static void read_int(int *arr, int n) {
    int i;
    for (i = 0; i < n; i++) { 
        scanf("%d", &arr[i]);
    }
}

// C = A + B
static void mat_add(int *A, int *B, int *C, int n) {
    int i;
    for (i = 0; i < n; i++) { 
        C[i] = A[i] + B[i];
    }
}

// zero row count
static int zero_row_count(int *C, int m, int n) {
    int sum, i, j, count = 0;
    for (i = 0; i < m; i++)  {
        sum = 0;
        for (j = 0; j < n; j++)  { 
            sum += C[i*n+j];
        }
        if (sum == 0) count++;
    }
    return count;
}

// zero col count
static int zero_col_count(int *C, int m, int n) { 
    int sum, i, j, count = 0;
    for (i = 0; i < n; i++)  {
        sum = 0;
        for (j = 0; j < m; j++)  { 
            sum += C[j*n+i];
        }
        if (sum == 0) count++;
    }
    return count;
}

int main(int argc, char** argv) {
    int m, n;               // 矩阵行列
    int *A, *B, *C;         // 矩阵 A，B 矩阵的和 C

    int total_zero;         // 0 行加 0 列的个数

    while (scanf("%d %d\n", &m, &n) != EOF) {
        if (m == 0) break;

        A = (int*)malloc(m * n * sizeof(int));
        B = (int*)malloc(m * n * sizeof(int));
        C = (int*)malloc(m * n * sizeof(int));

        read_int(A, m*n);
        read_int(B, m*n);

        mat_add(A, B, C, m*n);

        total_zero = zero_row_count(C, m, n);
        total_zero += zero_col_count(C, m, n);

        printf("%d\n", total_zero);

        free(A);
        free(B);
        free(C);
    }

    return 0;
}
