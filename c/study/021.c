// 40个3和30个2凑100
#include <stdio.h>

int main(int argc, char *argv[])
{
    int arr[][2] = {40, 3, 30, 2}; 
    int target = 100;
    int kinds = sizeof(arr)/sizeof(int)/2; 

    int i, j;
    for (i = 0; i < kinds; ++i) {
        for (j = 0;  j< arr[i][0]; j++) {
            if (target < arr[i][1]) break;
            target -= arr[i][1];
            //printf("debug: %d %d %d\n", target, arr[i][1], j);
            if (target < arr[i][1]) break;
        }
    }

    printf("%s\n", target == 0 ? "能凑够" : "不能凑够");
    return 0;
}
