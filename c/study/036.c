#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>

#define MAX 50 
static char line[MAX];
static int pos;


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

static struct Token current_token;

void back(){
    pos--;
}

char next_ch(){
    return pos >= MAX ? '\0': line[pos++];
}

char *parse_str() {
    static char buf[MAX];
    char *p = buf, c;
    while (c = next_ch()) {
        if (isalpha(c) || isdigit(c)) {
            *p++ = c;
        } else {
            break;
        }
    }
    *p = '\0';
    return buf;
}


int parse_int() {
    static char buf[MAX];
    char *p = buf, c;
    while (c = next_ch()) {
        if (isdigit(c)) {
            *p++ = c;
        } else {
            break;
        }
    }
    *p = '\0';
    return atoi(buf);
}

struct Token *token(){
    char c;
    while ((c = next_ch()) == ' ') ;

    if (isalpha(c)) {
        current_token.type = TYPE_ID; 
        back();
        current_token.data.s= parse_str(); 
        return &current_token;
    }

    if (isdigit(c)) {
        current_token.type = TYPE_NUM; 
        back();
        current_token.data.n= parse_int(); 
        return &current_token;
    }

    return NULL;
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
    char *s = "abc 111";
    strncpy(line, s, strlen(s)+1);
    pos = 0;
    tokens();
}

int main()
{
    test_tokens();
    return 0;
}
