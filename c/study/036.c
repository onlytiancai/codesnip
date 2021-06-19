#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>

#define MAX 50 
static char line[MAX];
static int pos;


struct Token {
    enum TokenType {TYPE_ID, TYPE_NUM, TYPE_GE, TYPE_PLUS, TYPE_MINUS, TYPE_STAR, TYPE_SLASH } type;
    union data{ int n; char c;} data; 
};
const char* typeNames[] = {"id", "num", "ge", "plus", "minus", "star", "slash"};
void print_token_id(struct Token *t) { printf("type=%s data=%c\n", typeNames[t->type], t->data.c); }
void print_token_num(struct Token *t) { printf("type=%s data=%d\n", typeNames[t->type], t->data.n); }           
void print_token_other(struct Token *t) { printf("type=%s \n", typeNames[t->type]); }           

static void (* const pf[])(struct Token *t) = {
    print_token_id, print_token_num, print_token_other, print_token_other,
    print_token_other, print_token_other, print_token_other
};
void print_token(struct Token *t){ pf[t->type](t); }


void back(){ pos--; }
char next(){ return pos >= MAX ? '\0': line[pos++]; }

struct Token *token(){
    static struct Token token;
    char c;
    while ((c = next()) == ' ') ;

    if (isalpha(c)) {
        token.type = TYPE_ID; 
        token.data.c = c; 
        return &token;
    }

    if (isdigit(c)) {
        token.type = TYPE_NUM; 
        token.data.n = c - '0'; 
        return &token;
    }

    if (c == '=') {
        token.type = TYPE_GE; 
        return &token;
    }

    if (c == '\0') return NULL; 

    fprintf(stderr, "unknow char %c\n", c);
    exit(EXIT_FAILURE);
}



int tokens(){
    struct Token *t;
    while(t=token()){
        print_token(t);
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

void test_tokens(){
    char *s = "a = 3";
    strncpy(line, s, strlen(s)+1);
    pos = 0;
    tokens();
}

int main()
{
    test_tokens();
    return 0;
}
