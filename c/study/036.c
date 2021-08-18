#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>

#define MAX 50 
#define MAX_TOKEN_LEN 10 
#define MAX_NODES 100 
#define MAX_TOKENS 100 
static char line[MAX];
static int char_index = 0;
void back_char() { char_index--; }
char next_char() { return char_index >= MAX ? '\0': line[char_index ++]; }

enum TokenType {TYPE_ID, TYPE_NUM, TYPE_EQ, TYPE_PLUS, TYPE_MINUS, TYPE_STAR, TYPE_SLASH };
struct Token {
    enum TokenType type;
    union data{ int n; char c;} data; 
};
const char* typeNames[] = {"id", "num", "ge", "+", "-", "*", "/"};
void print_token_id(struct Token *t) { printf("type=%s data=%c\n", typeNames[t->type], t->data.c); }
void print_token_num(struct Token *t) { printf("type=%s data=%d\n", typeNames[t->type], t->data.n); }           
void print_token_other(struct Token *t) { printf("type=%s \n", typeNames[t->type]); }           

static void (* const pf[])(struct Token *t) = {
    print_token_id, print_token_num, print_token_other, print_token_other,
    print_token_other, print_token_other, print_token_other
};
void print_token(struct Token *t){ pf[t->type](t); }

struct Token token_list[MAX_TOKENS];
static int token_alloc_index = 0;
static int token_index = 0;
struct Token *Token_new() {
    if (token_alloc_index >= MAX_TOKENS) {
        fprintf(stderr, "too many tokens\n");
        exit(EXIT_FAILURE);
    }
    return &token_list[token_alloc_index++];
}

void back_token() {token_index--;};
struct Token *next_token() {
    if (token_index >= token_alloc_index-1) {
        return NULL;
    }
    //printf("000 ");
    //print_token(&token_list[token_index]);
    return &token_list[token_index++];
}

struct Token *token(){
    struct Token *token = Token_new();
    char char_table[] = {'=', '+', '-', '*', '/'};
    enum TokenType type_table[] = {TYPE_EQ, TYPE_PLUS, TYPE_MINUS, TYPE_STAR, TYPE_SLASH};
    char c, buf[MAX_TOKEN_LEN], *p = buf;
    int i;

    while ((c = next_char()) == ' ') ;

    if (isalpha(c)) {
        token->type = TYPE_ID; 
        token->data.c = c; 
        return token;
    }

    if (isdigit(c)) {
        token->type = TYPE_NUM; 
        do { *p++ = c; }
        while(isdigit(c = next_char()));
        *p = '\0';
        token->data.n = atoi(buf); 
        return token;
    }

    for (i = 0; i < sizeof(char_table) / sizeof(char_table[0]); ++i) {
       if (c == char_table[i]) {
           token->type = type_table[i]; 
           return token;
       } 
    }

    if (c == '\0') return NULL; 

    fprintf(stderr, "unknow char %c\n", c);
    exit(EXIT_FAILURE);
}


int tokens(){
    struct Token *t;
    while(t=token()){
    }
}


void repl() {
    while (1) {
        fgets(line, MAX, stdin);
        line[strcspn(line, "\n")] = 0;
        if (strcmp(line, "quit") == 0) {
            break;
        }

        tokens();
    }
}


struct ASTNode {
    struct Token *token;
    struct ASTNode *left;
    struct ASTNode *right;
};

static struct ASTNode node_list[MAX_NODES];
static int node_index = 0;
struct ASTNode *ASTNode_new() {
    if (node_index >= MAX_NODES) {
        fprintf(stderr, "too many ast nodes\n");
        exit(EXIT_FAILURE);
    }
    return &node_list[node_index++];
}
void print_node(struct ASTNode *node, int level) {
    if (node == NULL) return;
    int i = 0;
    for (i = 0; i < level; ++i) {
       printf("\t"); 
    }
    if (node->token) print_token(node->token); 
    print_node(node->left, level+1);
    print_node(node->right, level+1);
}
// pri -> Id | Literal | (exp)
struct ASTNode *match_pri() {
    struct Token *t; 
    struct ASTNode *ret;
    t = next_token();
    if (t != NULL && t->type == TYPE_ID || t->type == TYPE_NUM) {
        ret = ASTNode_new();
        ret->token = t;
        return ret;
    }

    back_token();
    return NULL;
}

// mul -> pri | mul * pri 
// mul -> pri (* pri)*
struct ASTNode *match_mul() {
    struct Token *t; 
    struct ASTNode *ret, *left, *right;

    if ( (left = match_pri() ) == NULL) return NULL;

    ret = left;
    while (1) {
        t = next_token();
        if (t == NULL || t->type != TYPE_STAR) {
            back_token();
            break;
        } 
        ret = ASTNode_new();
        ret->token = t;

        right = match_pri();
        if (right == NULL) {
            fprintf(stderr, "expect pri\n");
            exit(EXIT_FAILURE);
        }
        ret->left = left;
        ret->right = right;
        left = ret;
    }

    return ret;

}
// add -> mul | add + mul 
// add -> mul add'
// add' -> + mul add' | ε
// add -> mul (+ mul)* 
struct ASTNode *match_add() {
    struct Token *t; 
    struct ASTNode *ret, *left, *right;

    if ( (left = match_mul() ) == NULL) return NULL;
    ret = left;
    while (1) {
        t = next_token();

        if (t == NULL || t->type != TYPE_PLUS) {
            back_token();
            break;
        } 

        ret = ASTNode_new();
        ret->token = t;

        right = match_mul();
        if (right == NULL) {
            fprintf(stderr, "expect mul\n");
            exit(EXIT_FAILURE);
        }
        ret->left = left;
        ret->right = right;
        left = ret;
    }

    return ret;
}


struct ASTNode *match_exp() {
    return match_add();
}

// declare -> id = exp
struct ASTNode *match_declare() {
    struct ASTNode *ret, *left, *right;;
    struct Token *t; 

    t = next_token();
    if (! (t != NULL && t->type == TYPE_ID) ) {
        back_token();
        return NULL;
    }

    left = ASTNode_new();
    left->token = t;

    t = next_token();
    if (! (t != NULL && t->type == TYPE_EQ) ) {
        fprintf(stderr, "expect =\n");
        exit(EXIT_FAILURE);
    }
    ret = ASTNode_new();
    ret->token = t;

    if ((right = match_exp()) == NULL) {
        fprintf(stderr, "expect exp\n");
        exit(EXIT_FAILURE);
    }

    ret->left = left;
    ret->right = right;

    return ret;
}

void test_tokens(){
    printf("## test tokens\n");
    char *s = "a = 321 + b * 777";
    printf("%s\n", s);
    strncpy(line, s, strlen(s)+1);
    char_index = 0;
    tokens();

    struct Token *t;
    while(t=next_token()){
        print_token(t);
    }
}


void test_match() {
    printf("## test match\n");
    char *s = "a = 3 * 2 + 4 * 5 * 6 + 3 + 7";
    strncpy(line, s, strlen(s)+1);
    printf("%s\n", s);
    char_index = 0;
    token_alloc_index = 0;
    token_index = 0;
    tokens();
    print_node(match_declare(), 0);
}

int main()
{
    test_tokens();
    test_match();
    return 0;
}
