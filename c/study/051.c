// gcc -fsanitize=address -fno-omit-frame-pointer -O1 -g 051.c -o use-after-free
// https://zhuanlan.zhihu.com/p/375122996
// \time -f 'time cost=%es' grep -E '^781972,' test_data.csv 
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <limits.h>

#define DEFAULT_SIZE 100000

struct Node {
    // val, priority,count,pos
    int v,p,n,pos;
    int l, r;
};

static struct Node *nodes;

static int cur_idx = 0;
static int max_idx = DEFAULT_SIZE;
static void nodes_init() { 
    nodes = (struct Node *)malloc(DEFAULT_SIZE*sizeof(struct Node)); 
}

static void nodes_extend() { 
    struct Node n;
    for (int i = 0; i < cur_idx; ++i) {
        n = nodes[i];
    }
    printf("nodes extend %d %d %p\n", cur_idx, (cur_idx+DEFAULT_SIZE), nodes);
    struct Node *newnodes = malloc((cur_idx+DEFAULT_SIZE)*sizeof(struct Node)); 
    memcpy(newnodes, nodes, cur_idx*sizeof(struct Node));
    // free(nodes);
    nodes = newnodes;
    for (int i = 0; i < cur_idx; ++i) {
        n = nodes[i];
        //printf("444: i=%d v=%d p=%d n=%d pos=%d l=%d r=%d\n", i, n.v, n.p, n.n,n.pos,n.l,n.r);
    }

    max_idx = cur_idx+DEFAULT_SIZE;
    if (nodes == NULL) {
        perror("realloc, error");
        exit(EXIT_FAILURE);
    }
}



int new_node(int v, int pos) { 
    if (cur_idx >= max_idx) {
        nodes_extend();
    }
    int cur = cur_idx;
    struct Node *p = &nodes[cur];
    p->v = v;
    p->l  = -1;
    p->r = -1;
    p->n = 1;
    p->pos = pos;
    p->p = rand();
    cur_idx++;
    return cur;
}

void print_tree(int inode, int indent)
{
    if (inode == -1) return;
    struct Node node = nodes[inode];
    printf("|");
    for (int i = 0; i < indent; ++i) printf("--|");
    printf("%d\n", node.v);
    if (node.l != -1) print_tree(node.l, indent+1);
    if (node.r != -1) print_tree(node.r, indent+1);
}


void right_rotate(int *ia) {
    //Tree b = a->lson;
    //a->lson = b->rson, b->rson = a, a = b;

    struct Node *a = &nodes[*ia];
    int ib = a->l;
    struct Node *b = &nodes[ib];
    a->l = b->r, b->r = *ia, *ia = ib;
}

void left_rotate(int *ia) {
    struct Node *a = &nodes[*ia];
    int ib = a->r;
    struct Node *b = &nodes[ib];
    a->r = b->l, b->l = *ia, *ia = ib;
}

void insert(int *irt, int val, int pos) {
    struct Node *old_nodes = nodes;
    int new_inode;
    if (*irt == -1) {
        new_inode = new_node(val, pos);
        if (old_nodes == nodes) {
            *irt = new_inode;
        } else {
            // 计算新的指针位置
            uintptr_t diff = (uintptr_t)irt-(uintptr_t)old_nodes;
            int *new_irt = (int*)((uintptr_t)nodes+diff); 
            *new_irt = new_inode;
            irt = new_irt;
        }
    }
    struct Node *rt = &nodes[*irt];
    if (rt->v == val) rt->n++; // 已经有这个点了
    else if (rt->v > val) {
        // 如果这个节点的值大了就跑到左子树
        insert(&(rt->l), val, pos);
        rt = &nodes[*irt]; // 内存可能被迁移，要重新指向指针
        // 因为只更改了左子树，只用判断自己和左子树的优先级
        if (rt->p < nodes[rt->l].p) right_rotate(irt);
    }
    else {
        // 如果这个节点的值小了就跑到右子树
        insert(&(rt->r), val, pos);
        rt = &nodes[*irt]; // 内存可能被迁移，要重新指向指针
        if (rt->p < nodes[rt->r].p) left_rotate(irt);
    }
}

int query(int ip, int val) {
    int pos = -1;
    struct Node *p;
    while (ip != -1) {
        p = &nodes[ip];
        if (p->v == val) return p->pos;
        else if (p->v > val) ip = p->l;
        else ip = p->r;
    }
    return pos;
}

void test01()
{
    int iroot = new_node(2, -1);
    struct Node *root = &nodes[iroot];
    struct Node *p;
    root->l = new_node(1, -1);
    root->r = new_node(6, -1);
    p = &nodes[root->r];
    p->l = new_node(4, -1);
    p->r = new_node(9, -1);
    print_tree(iroot, 0);
    printf("----\n");

    right_rotate(&iroot);
    print_tree(iroot, 0);
    printf("----\n");

    left_rotate(&iroot);
    print_tree(iroot, 0);

}

int cmpfunc (const void * a, const void * b)
{
   return ( *(int*)a - *(int*)b );
}

int binarySearch(int arr[], int l, int r, int x)
{
    while (l <= r) {
        int m = l + (r - l) / 2;
        if (arr[m] == x)
            return m;
        if (arr[m] < x)
            l = m + 1;
        else
            r = m - 1;
    }
    return -1;
}

void test02()
{
    clock_t start, end;
    int x, ret, *ret2 = NULL;
    double time_cost;

    int arr[DEFAULT_SIZE];
    start = clock(); 
    for (int i = 0; i < DEFAULT_SIZE; ++i) arr[i] = rand() % (DEFAULT_SIZE*2);
    time_cost = (double)(clock()-start)/CLOCKS_PER_SEC*1000;
    printf("build array time cost:%f ms\n", time_cost);
    printf("array length=%d\n", DEFAULT_SIZE);

    int iroot = -1;
    start = clock(); 
    for (int i = 0; i < DEFAULT_SIZE; ++i) insert(&iroot, arr[i], i);
    time_cost = (double)(clock()-start)/CLOCKS_PER_SEC*1000;
    printf("build tree time cost:%f ms\n", time_cost);

    x = rand() % (DEFAULT_SIZE*2); 

    start = clock(); 
    ret = -1; for (int i = 0; i < DEFAULT_SIZE; ++i) if (arr[i] == x) {ret = i;break;};
    time_cost = (double)(clock()-start)/CLOCKS_PER_SEC*1000;
    printf("array search: %d %d, time cost=%f ms\n", x, ret, time_cost);

    start = clock(); 
    ret = query(iroot, x);
    time_cost = (double)(clock()-start)/CLOCKS_PER_SEC*1000;
    printf("tree search: %d %d, time cost=%f ms\n", x, ret, time_cost);

    start = clock(); 
    qsort(arr, DEFAULT_SIZE, sizeof(int), cmpfunc);
    time_cost = (double)(clock()-start)/CLOCKS_PER_SEC*1000;
    printf("qsort: time cost=%f ms\n", time_cost);

    start = clock(); 
    ret2 = (int*)bsearch(&x, arr, DEFAULT_SIZE, sizeof(int), cmpfunc);
    time_cost = (double)(clock()-start)/CLOCKS_PER_SEC*1000;
    printf("bsearch:%d %d time cost=%f ms\n", x, ret2 == NULL ? -1 : *ret2, time_cost);

}

int read_csv(char* file, int *iroot, int limit)
{
    FILE *fp;
    char line[128]; 

    fp = fopen(file, "r");
    if(fp == NULL) {
        perror("fopen error");
        exit(-1);
    }

    char *s = NULL;
    int i = 0, key, pos;
    char *str, *token;
    if (limit == -1) limit = INT_MAX;
    while (fgets(line, 128, fp) !=NULL && ++i < limit) {
        s = strchr(line, '\n');
        if (s) {
            *s = '\0';
        }

        pos =  (int)(ftell(fp)-strlen(line)-1);
        str = line;
        token = strtok(str, ",");
        if (token != NULL) {
            key = atoi(token);
            insert(iroot, key, pos);
        }
    }
    fclose(fp);
    return i;
}

void test03()
{
    int iroot = -1;
    clock_t start, end;
    double time_cost;
    int x, ret, total;
    FILE *fp;


    char line[100], *s;
    printf("Welcome to Wawa Database.\ninput 'buld index' to build index.\ninput n to query a random data.\ninput q to exit:\n");
    printf("> ");
    while(NULL != fgets(line, 100, stdin)){
        s = strchr(line, '\n');
        if (s) {
            *s = '\0';
        }

        if (strcmp(line, "q") == 0) break;
        if (strcmp(line, "build index") == 0) {
            start = clock(); 
            total = read_csv("test_data.csv", &iroot, -1);
            time_cost = (double)(clock()-start)/CLOCKS_PER_SEC*1000;
            printf("build tree done, total: %d, time cost:%f ms\n", total, time_cost);
            printf("> ");
            continue;
        } 

        x = rand() % (total*2); 
        start = clock(); 
        ret = query(iroot, x);
        time_cost = (double)(clock()-start)/CLOCKS_PER_SEC*1000;
        if (ret!=-1) {
            fp = fopen("test_data.csv", "r");
            if(fp == NULL) {
                perror("fopen error");
                exit(-1);
            }
            fseek( fp, ret, SEEK_SET);

           if (fgets(line, 100, fp) !=NULL) {
               printf("%s", line);
           }
            fclose(fp);
        }
        printf("tree search: %d %d, time cost=%f ms\n", x, ret, time_cost);
        printf("> ");
    }
}

int main(int argc, char *argv[]) {
    srand((unsigned int)time(NULL));
    nodes_init();
    test03();
    return 0;
}

