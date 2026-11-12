/*
 * MiniLang —— 抽象语法树
 *
 * 职责：分配 AST 节点、把语法树打印成带连接线的树。
 */
#include "minilang.h"

/* 节点池：所有 AST 节点都从这里分配，用完为止 */
static Node nodes[MAX_NODES];
static int  nnodes;

/* 创建一个节点。memset 清零非常重要：否则 switch 里没覆盖到的字段
   会是随机值，后面调试会非常痛苦 */
Node *node(NodeType ty, Token t) {
    if (nnodes >= MAX_NODES) fatal_at(t, "语法树节点超出上限");
    Node *n = &nodes[nnodes++];
    memset(n, 0, sizeof(*n));
    n->type = ty;
    n->tok  = t;
    return n;
}

/* 把一棵子树打印成带连接线的树。
   prefix 是"给子节点用的缩进前缀"，逐层往下加 "│   " 或 "    " */
void dump_ast(Node *n, const char *prefix) {
    if (!n) return;

    Node *kids[128];
    int   m = 0;

    switch (n->type) {
        case N_NUM:
            printf("数字 %g\n", n->num);
            return;
        case N_STR:
            printf("字符串 \"%s\"\n", n->text);
            return;
        case N_VAR:
            printf("变量 %s\n", n->text);
            return;
        case N_UNARY:
            printf("一元 %s\n", type_name((TokType)n->op));
            kids[m++] = n->a;
            break;
        case N_BINARY:
            printf("二元 %s\n", type_name((TokType)n->op));
            kids[m++] = n->a;
            kids[m++] = n->b;
            break;
        case N_CALL:
            printf("调用 %s\n", n->text);
            for (int i = 0; i < n->count; i++) kids[m++] = n->items[i];
            break;
        case N_BLOCK:
            printf("代码块\n");
            for (int i = 0; i < n->count; i++) kids[m++] = n->items[i];
            break;
        case N_LET:
            printf("定义 %s\n", n->text);
            kids[m++] = n->a;
            break;
        case N_ASSIGN:
            printf("赋值 %s\n", n->text);
            kids[m++] = n->a;
            break;
        case N_EXPR:
            printf("表达式语句\n");
            kids[m++] = n->a;
            break;
        case N_IF:
            printf("如果\n");
            kids[m++] = n->a;   /* 条件 */
            kids[m++] = n->b;   /* then 分支 */
            if (n->c) kids[m++] = n->c;   /* else 分支（可能没有） */
            break;
        case N_FOR:
            printf("循环\n");
            if (n->a) kids[m++] = n->a;   /* 初始化 */
            if (n->b) kids[m++] = n->b;   /* 条件 */
            if (n->c) kids[m++] = n->c;   /* 更新 */
            for (int i = 0; i < n->count; i++) kids[m++] = n->items[i];  /* 循环体 */
            break;
        case N_FN:
            printf("函数 %s(", n->text);
            for (int i = 0; i < n->nparams; i++)
                printf("%s%s", i ? ", " : "", n->params[i]);
            printf(")\n");
            kids[m++] = n->a;   /* 函数体 */
            break;
        case N_RETURN:
            printf("返回\n");
            if (n->a) kids[m++] = n->a;   /* 返回的表达式（可以没有） */
            break;
    }

    for (int i = 0; i < m; i++) {
        int last = (i == m - 1);
        char sub[256];
        snprintf(sub, sizeof(sub), "%s%s", prefix, last ? "    " : "│   ");
        printf("%s%s", prefix, last ? "└── " : "├── ");
        dump_ast(kids[i], sub);
    }
}
