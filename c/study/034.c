#include <stdio.h>

#define N 10

struct conn_data{
   int fd; 
   int inuse;
};

static struct conn_data conn_data_list[N];
static struct conn_data *conn_data_free_list[N];

static void conn_data_list_init(){
    int i;
    for (i = 0; i < N; ++i) {
       struct conn_data cd = conn_data_list[i];
       cd.fd = 0;
       cd.inuse = 0;
    }
}

static struct conn_data *conn_data_new(int fd) {
    int i;
    for (i = 0; i < N; ++i) {
        if (conn_data_list[i].inuse == 0) {
            printf("new cd index:%d\n", i);
            conn_data_list[i].fd = fd;
            conn_data_list[i].inuse = 1;
            return &conn_data_list[i];
        }
    }
    return NULL;
}

int main()
{
    conn_data_list_init(); 
    int i;
    struct conn_data * p_cd;
    for (i = 0; i < N + 1; ++i) {
        p_cd = conn_data_new(i);
        if (p_cd != NULL) {
            printf("cd->fd: %d\n", p_cd->fd);
        } else {
            printf("new cd faild\n");
        }
    }
    return 0;
}
