#include <stdio.h>
#include <math.h>
// 目标函数 f
float f(float x, float n) { return x*x-n; }
// 目标函数的导数 f' 
float fd(float x) { return 2*x; }
int main(int argc, char *argv[]) {
    float y = 2;
    // built-in function
    printf("built-in function: %f\n", sqrt(y)); 

    // binary search
    float l = 0, r = y, x;
    int i = 0; // 防止死循环
    do {
        x = (l+r)/y;
        printf("debug binary search: %d %f %f\n", i, x, fabs(x*x - y));
        if (x*x > y) { r = x; }
        else { l = x; } 
    } while(fabs(x*x - y) > 0.0001 && ++i < 100); 
    printf("binary search: %f\n", x); 

    // Newton Iteration 
    float x0 = 0; 
    float x1 = y;
    i = 0; // 防止不能收敛
    do {
        printf("debug newton iter: %d %f %f\n", i, x0, x1);
        // x0 初始化为一个随机的点，后续为新的 x1 点 
        x0 = x1; 
        // x1 为 f 函数在 x0 的切线(导数)与 x 轴的交叉点
        // 由点斜式得到直线方程 y-y0=k0(x-x0)，令 y=0，求 x
        // 先两边除以 k0，再把 x 提到左边，
        // y-y0=k0(x-x0)  => y0/k0=x-x0 => x=x0-y0/k0
        // 其中 y0 为函数在 x0 的值，k0 为 x0 处的导数
        x1 = x0 - f(x0, y) / fd(x0);
    // 当 x0 和 x1 足够接近时表示已经收敛，结束迭代
    } while (fabs(x0-x1) > 0.0001 && ++i < 100);
    printf("newton iter: %f\n", x1); 

    return 0;
}
