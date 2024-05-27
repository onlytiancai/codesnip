#include <stdio.h>
struct data { char a[1272]; char b; unsigned char *c;};
void print_data(struct data *data) {
    printf("print_data:%d %d\n", data->b, (int)*(data->c));
}
int main(int argc, char *argv[])
{
    unsigned char c = 200;
    struct data data;
    data.b = 100;
    data.c = &c;
    printf("%p %p %ld\n", &data, &(data.b), (long)&(data.b)-(long)&data);
    print_data(&data);
    return 0;
}
