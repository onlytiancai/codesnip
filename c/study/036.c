#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>

#define MAX 50 
static char line[MAX];
static int pos;

static struct Token {
    char type;
    char *str_value;
    int int_value;
} current_token;

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
        current_token.type = 'a'; 
        back();
        current_token.str_value = parse_str(); 
        return &current_token;
    }

    if (isdigit(c)) {
        current_token.type = 'd'; 
        back();
        current_token.int_value = parse_int(); 
        return &current_token;
    }

    return NULL;
}

int tokens(){
    struct Token *t;
    pos = 0;
    while(t=token()){
        printf("%c %s %d\n", t->type, t->str_value, t->int_value);
    }
}

int main()
{
    while (1) {
        fgets(line, MAX, stdin);
        line[strcspn(line, "\n")] = 0;
        if (strcmp(line, "quit") == 0) {
            break;
        }

        tokens();
    }
    return 0;
}
