// np.array([[1, 2, 3],[4, 5, 6]])
// b = np.array([[2, 4, 1, 2],[6, 8, 3, 4],[3, 5, 5, 6]])
// a.dot(b)
// array([
//  [2, 4, 1, 2],
//  [6, 8, 3, 4],
//  [3, 5, 5, 6]]
// )
#include <stdio.h>

#define aM  2
#define aN  3
#define bM  3
#define bN  4

int main(int argc, char* argv[]) {
    printf("hello 009\n");

    int a[aM][aN] = {
        {1, 2, 3},
        {4, 5, 6},
    };

    int b[bM][bN] = {
        {2, 4, 1, 2},
        {6, 8, 3, 4},
        {3, 5, 5, 6}
    };

    int i, j, k,  c[aM][bN];
    for (i = 0; i < aM; i++) {
        for (j = 0; j < bN; j++) {
            c[i][j] = 0;
            for (k = 0; k < aN; k++) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    for (i = 0; i < aM; i++) {
        for (j = 0; j < bN; j++) {
            printf("%d ", c[i][j]);
        }
        printf("\n");
    }

    return 0;
}
