#include <stdio.h>
#include <time.h>

int main() {
    printf("hello 014\n");
    srand(time(NULL));
    int years[] = {2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019};
    int year = years[rand() % 10];
    printf("year=%d\n", year);

    if (year % 4 == 0) {
        if (year % 100 == 0) {
            printf(year % 400 == 0 ? "是闰年\n" : "不是闰年\n");
        } else {
            printf("是闰年\n");
        }
    } else {
        printf("不是闰年\n");
    }

    return 0;
}
