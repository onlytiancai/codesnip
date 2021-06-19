#include <stdio.h>

struct Token {
    enum TokenType {TYPE_ID, TYPE_NUM, TYPE_GE, TYPE_PLUS, TYPE_MINUS, TYPE_STAR, TYPE_SLASH } type;
    union data{ int n; char c; char *s; } data; 
};
const char* typeNames[] = {"id", "num", "ge", "plus", "minus", "star", "slash"};
void print_token_id(struct Token *t) { printf("type=%s data=%s\n", typeNames[t->type], t->data.s); }
void print_token_num(struct Token *t) { printf("type=%s data=%d\n", typeNames[t->type], t->data.n); }           
void print_token_other(struct Token *t) { printf("type=%s ", typeNames[t->type]); }           

static void (* const pf[])(struct Token *t) = {
    print_token_id, print_token_num, print_token_other, print_token_other,
    print_token_other, print_token_other, print_token_other
};
void print_token(struct Token *t){ pf[t->type](t); }


int main(int argc, char *argv[])
{
    struct Token tokens[] = {
        {TYPE_ID, {.s = "int"}},
        {TYPE_ID, {.s = "a"}},
        {TYPE_GE, {.c = '='}},
        {TYPE_NUM, {.n = 3}},
    };
    int i;
    for (i = 0; i < sizeof(tokens) / sizeof(struct Token); ++i) {
       print_token(&tokens[i]); 
    }
    return 0;
}
