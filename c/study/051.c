#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define MAX 100

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

void function_name()
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

int main(int argc, char *argv[]) {
    srand((unsigned int)time(NULL));

    struct Node *root = NULL;
    for (int i = 0; i < 100; ++i) {
        insert(&root, i);
    }
    print_tree(root, 0);
    return 0;
}
