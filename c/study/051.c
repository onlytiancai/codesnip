#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <limits.h>
#define MAX 1000000

static struct Node {
    int v,p,n;
    struct Node *l, *r;
} nodes[MAX];

static int cur_idx = 0;

static struct Node *new_node(int v) { 
    int cur = cur_idx;
    nodes[cur].v = v;
    nodes[cur].l = NULL;
    nodes[cur].r = NULL;
    nodes[cur].n = 1;
    nodes[cur].p = rand();
    cur_idx++;
    return &nodes[cur];
}

void print_tree(struct Node *node, int indent)
{
    printf("|");
    for (int i = 0; i < indent; ++i) printf("--|");
    if (node != NULL) printf("%d\n", node->v);
    if (node->l != NULL) print_tree(node->l, indent+1);
    if (node->r != NULL) print_tree(node->r, indent+1);
}


void right_rotate(struct Node **a) {
    struct Node *b = (*a)->l;
    (*a)->l = b->r, b->r = *a, *a = b;
}

void left_rotate(struct Node **a) {
    struct Node *b = (*a)->r;
    (*a)->r = b->l, b->l = *a, *a = b;
}

void insert(struct Node **rt, int val) {
    if (*rt == NULL) {*rt = new_node(val); return;}
    if ((*rt)->v == val) (*rt)->n++; // 已经有这个点了
    else if ((*rt)->v > val) {
        // 如果这个节点的值大了就跑到左子树
        insert(&((*rt)->l), val);
        // 因为只更改了左子树，只用判断自己和左子树的优先级
        if ((*rt)->p < (*rt)->l->p) right_rotate(rt);
    }
    else {
        // 如果这个节点的值小了就跑到右子树
        insert(&((*rt)->r), val);
        if ((*rt)->p < (*rt)->r->p) left_rotate(rt);
    }
}

int query(struct Node *p, int val) {
    int rank = -1;
    while (p != NULL)
        if (p->v == val) return p->p;
        else if (p->v > val) p = p->l;
        else p = p->r;
    return rank;
}

void test01()
{
    struct Node *root = new_node(2);
    root->l = new_node(1);
    root->r = new_node(6);
    root->r->l = new_node(4);
    root->r->r = new_node(9);
    print_tree(root, 0);

    right_rotate(&root);
    print_tree(root, 0);

    left_rotate(&root);
    print_tree(root, 0);

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

    int arr[MAX];
    start = clock(); 
    for (int i = 0; i < MAX; ++i) arr[i] = rand() % (MAX*2);
    time_cost = (double)(clock()-start)/CLOCKS_PER_SEC*1000;
    printf("build array time cost:%f ms\n", time_cost);
    printf("array length=%d\n", MAX);

    struct Node *root = NULL;
    start = clock(); 
    for (int i = 0; i < MAX; ++i) insert(&root, arr[i]);
    time_cost = (double)(clock()-start)/CLOCKS_PER_SEC*1000;
    printf("build tree time cost:%f ms\n", time_cost);

    x = rand() % (MAX*2); 

    start = clock(); 
    ret = -1; for (int i = 0; i < MAX; ++i) if (arr[i] == x) {ret = i;break;};
    time_cost = (double)(clock()-start)/CLOCKS_PER_SEC*1000;
    printf("array search: %d %d, time cost=%f ms\n", x, ret, time_cost);

    start = clock(); 
    ret = query(root, x);
    time_cost = (double)(clock()-start)/CLOCKS_PER_SEC*1000;
    printf("tree search: %d %d, time cost=%f ms\n", x, ret, time_cost);

    start = clock(); 
    qsort(arr, MAX, sizeof(int), cmpfunc);
    time_cost = (double)(clock()-start)/CLOCKS_PER_SEC*1000;
    printf("qsort: time cost=%f ms\n", time_cost);

    start = clock(); 
    ret2 = (int*)bsearch(&x, arr, MAX, sizeof(int), cmpfunc);
    time_cost = (double)(clock()-start)/CLOCKS_PER_SEC*1000;
    printf("bsearch:%d %d time cost=%f ms\n", x, ret2 == NULL ? -1 : *ret2, time_cost);

}

void read_csv(char* file, struct Node **root, int limit)
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
            insert(root, key);
        }
    }

    fclose(fp);
}

int main(int argc, char *argv[]) {
    srand((unsigned int)time(NULL));

    struct Node *root = NULL;
    clock_t start, end;
    double time_cost;

    start = clock(); 
    read_csv("/data/test_data.csv", &root, -1);
    time_cost = (double)(clock()-start)/CLOCKS_PER_SEC*1000;
    printf("build tree time cost:%f ms\n", time_cost);
    // print_tree(root, 0);

    return 0;
}

