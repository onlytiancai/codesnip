/*
 * MiniLang - 单文件小型解释器
 * 编译: cc -std=c11 -Wall -Wextra -O2 minilang.c -lm -o minilang
 * 运行: ./minilang demo.ml
 *
 * 语法:
 *   let x = 10;
 *   x = x + 1;
 *   if (x > 10) { print("x={x}\n"); } else { print("small\n"); }
 *   for (let i = 0; i < 5; i = i + 1) { print("{i}\n"); }
 *   fn add(a, b) { return a + b; }
 *
 * 为保持单文件和易读性，字符串只用于 print 模板；表达式值为数字。
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <stdarg.h>

#define MAX_TOKENS 20000
#define MAX_NODES  20000
#define MAX_VARS   2048
#define MAX_FUNCS  256
#define MAX_PARAMS 32
#define MAX_CALL_DEPTH 128
#define MAX_LOOP_STEPS 1000000
#define MAX_TEXT 4096

typedef enum {
    T_EOF, T_NUM, T_ID, T_STR,
    T_LET, T_FN, T_RETURN, T_IF, T_ELSE, T_FOR,
    T_LP, T_RP, T_LB, T_RB, T_COMMA, T_SEMI,
    T_PLUS, T_MINUS, T_STAR, T_SLASH, T_PERCENT,
    T_ASSIGN, T_EQ, T_NE, T_LT, T_LE, T_GT, T_GE,
    T_AND, T_OR, T_NOT
} TokType;

typedef struct {
    TokType type;
    char text[MAX_TEXT];
    double num;
    int line, col;
} Token;

static Token toks[MAX_TOKENS];
static int ntok, pos;
static const char *src;
static int src_line, src_col;
static int had_error;

static void error_at(int line, int col, const char *fmt, ...) {
    va_list ap;
    fprintf(stderr, "错误 [%d:%d]: ", line, col);
    va_start(ap, fmt); vfprintf(stderr, fmt, ap); va_end(ap);
    fputc('\n', stderr);
    had_error = 1;
}
static void fatal_at(Token t, const char *fmt, ...) {
    va_list ap;
    fprintf(stderr, "错误 [%d:%d]: ", t.line, t.col);
    va_start(ap, fmt); vfprintf(stderr, fmt, ap); va_end(ap);
    fputc('\n', stderr);
    had_error = 1;
    exit(1);
}
static char *dupstr(const char *s) { size_t n = strlen(s) + 1; char *p = (char *)malloc(n); if (!p) { perror("malloc"); exit(1); } memcpy(p, s, n); return p; }
static int is_id_start(int c) { return isalpha((unsigned char)c) || c == '_'; }
static int is_id_char(int c) { return isalnum((unsigned char)c) || c == '_'; }

static void add_token(TokType type, const char *text, double num, int line, int col) {
    if (ntok >= MAX_TOKENS) { error_at(line, col, "Token 数量超出上限"); exit(1); }
    Token *t = &toks[ntok++];
    t->type = type; t->num = num; t->line = line; t->col = col;
    snprintf(t->text, sizeof(t->text), "%s", text ? text : "");
}
static void lex(void) {
    ntok = 0; src_line = 1; src_col = 1;
    size_t i = 0;
    while (src[i]) {
        int c = (unsigned char)src[i];
        if (c == ' ' || c == '\t' || c == '\r') { i++; src_col++; continue; }
        if (c == '\n') { i++; src_line++; src_col = 1; continue; }
        if (c == '/' && src[i+1] == '/') {
            while (src[i] && src[i] != '\n') { i++; src_col++; }
            continue;
        }
        int line = src_line, col = src_col;
        if (isdigit(c) || (c == '.' && isdigit((unsigned char)src[i+1]))) {
            size_t start = i;
            int dots = 0;
            while (isdigit((unsigned char)src[i]) || src[i] == '.') {
                if (src[i] == '.' && ++dots > 1) break;
                i++; src_col++;
            }
            char b[128]; size_t n = i-start;
            if (n >= sizeof(b)) fatal_at((Token){.line=line,.col=col}, "数字字面量太长");
            memcpy(b, src+start, n); b[n] = 0;
            char *end = NULL; double v = strtod(b, &end);
            if (!end || *end) fatal_at((Token){.line=line,.col=col}, "非法数字 '%s'", b);
            add_token(T_NUM, b, v, line, col); continue;
        }
        if (is_id_start(c)) {
            size_t start = i;
            while (is_id_char((unsigned char)src[i])) { i++; src_col++; }
            char b[128]; size_t n = i-start;
            if (n >= sizeof(b)) fatal_at((Token){.line=line,.col=col}, "标识符太长");
            memcpy(b, src+start, n); b[n] = 0;
            TokType ty = T_ID;
            if (!strcmp(b,"let")) ty=T_LET; else if (!strcmp(b,"fn")) ty=T_FN;
            else if (!strcmp(b,"return")) ty=T_RETURN; else if (!strcmp(b,"if")) ty=T_IF;
            else if (!strcmp(b,"else")) ty=T_ELSE; else if (!strcmp(b,"for")) ty=T_FOR;
            add_token(ty,b,0,line,col); continue;
        }
        if (c == '"') {
            i++; src_col++; char b[MAX_TEXT]; size_t n=0; int closed=0;
            while (src[i]) {
                if (src[i] == '"') { i++; src_col++; closed=1; break; }
                if (src[i] == '\n') fatal_at((Token){.line=line,.col=col}, "字符串不能直接跨行，请使用 \\n");
                char ch = src[i++]; src_col++;
                if (ch == '\\' && src[i]) {
                    char e = src[i++]; src_col++;
                    if (e=='n') ch='\n'; else if (e=='t') ch='\t';
                    else if (e=='r') ch='\r'; else if (e=='"' || e=='\\') ch=e;
                    else fatal_at((Token){.line=line,.col=col}, "不支持的字符串转义 \\%c", e);
                }
                if (n+1 >= sizeof(b)) fatal_at((Token){.line=line,.col=col}, "字符串太长");
                b[n++]=ch;
            }
            if (!closed) fatal_at((Token){.line=line,.col=col}, "字符串缺少结束引号");
            b[n]=0; add_token(T_STR,b,0,line,col); continue;
        }
        TokType ty = T_EOF; char two[3] = {0,0,0}; two[0]=(char)c; two[1]=src[i+1];
        if (!strncmp(two,"==",2)) ty=T_EQ;
        else if (!strncmp(two,"!=",2)) ty=T_NE;
        else if (!strncmp(two,"<=",2)) ty=T_LE;
        else if (!strncmp(two,">=",2)) ty=T_GE;
        else if (!strncmp(two,"&&",2)) ty=T_AND;
        else if (!strncmp(two,"||",2)) ty=T_OR;
        if (ty != T_EOF) { i+=2; src_col+=2; add_token(ty,two,0,line,col); continue; }
        switch(c) {
            case '(': ty=T_LP; break; case ')': ty=T_RP; break;
            case '{': ty=T_LB; break; case '}': ty=T_RB; break;
            case ',': ty=T_COMMA; break; case ';': ty=T_SEMI; break;
            case '+': ty=T_PLUS; break; case '-': ty=T_MINUS; break;
            case '*': ty=T_STAR; break; case '/': ty=T_SLASH; break;
            case '%': ty=T_PERCENT; break; case '=': ty=T_ASSIGN; break;
            case '<': ty=T_LT; break; case '>': ty=T_GT; break;
            case '!': ty=T_NOT; break;
            default: fatal_at((Token){.line=line,.col=col}, "无法识别字符 '%c'", c);
        }
        i++; src_col++; add_token(ty,two,0,line,col);
    }
    add_token(T_EOF,"<文件结束>",0,src_line,src_col);
}

typedef enum { N_NUM, N_STR, N_VAR, N_UNARY, N_BINARY, N_CALL,
               N_BLOCK, N_LET, N_ASSIGN, N_EXPR, N_IF, N_FOR,
               N_FN, N_RETURN } NodeType;
typedef struct Node Node;
struct Node {
    NodeType type;
    Token tok;
    double num;
    char text[MAX_TEXT];
    int op;
    Node *a,*b,*c;
    Node *items[128]; int count;
    char *params[MAX_PARAMS]; int nparams;
};
static Node nodes[MAX_NODES]; static int nnodes;
static Node *node(NodeType ty, Token t) {
    if (nnodes >= MAX_NODES) fatal_at(t,"语法树节点超出上限");
    Node *n=&nodes[nnodes++]; memset(n,0,sizeof(*n)); n->type=ty; n->tok=t; return n;
}
static Token *cur(void) { return &toks[pos]; }
static int match(TokType t) { if (cur()->type==t) { pos++; return 1; } return 0; }
static Token expect(TokType t, const char *what) {
    if (cur()->type != t) fatal_at(*cur(),"语法错误：%s，当前是 '%s'",what,cur()->text);
    return toks[pos++];
}
static Node *expression(void);
static Node *statement(void);
static Node *block(void);

static Node *primary(void) {
    Token t=*cur();
    if (match(T_NUM)) { Node*n=node(N_NUM,t); n->num=t.num; return n; }
    if (match(T_STR)) { Node*n=node(N_STR,t); snprintf(n->text,sizeof(n->text),"%s",t.text); return n; }
    if (match(T_ID)) {
        if (match(T_LP)) {
            Node*n=node(N_CALL,t); snprintf(n->text,sizeof(n->text),"%s",t.text);
            if (cur()->type != T_RP) {
                do {
                    if(n->count>=128) fatal_at(*cur(),"函数参数过多");
                    n->items[n->count++]=expression();
                } while(match(T_COMMA));
            }
            expect(T_RP,"函数调用需要 ')'");
            return n;
        }
        Node*n=node(N_VAR,t); snprintf(n->text,sizeof(n->text),"%s",t.text); return n;
    }
    if (match(T_LP)) { Node*n=expression(); expect(T_RP,"表达式需要 ')'"); return n; }
    fatal_at(t,"语法错误：此处需要数字、变量、函数调用或括号表达式");
    return NULL;
}
static Node *unary(void) {
    if (cur()->type==T_MINUS || cur()->type==T_PLUS || cur()->type==T_NOT) {
        Token t=*cur(); pos++; Node*n=node(N_UNARY,t); n->op=t.type; n->a=unary(); return n;
    }
    return primary();
}
static Node *bin_level(Node *(*next)(void), const TokType *ops, int nops) {
    Node *left=next();
    for (;;) {
        int found=0;
        for(int i=0;i<nops;i++) if(cur()->type==ops[i]) {
            Token t=*cur(); pos++; Node*n=node(N_BINARY,t); n->op=t.type; n->a=left; n->b=next(); left=n; found=1; break;
        }
        if(!found) return left;
    }
}
static Node *mul(void) { const TokType o[]={T_STAR,T_SLASH,T_PERCENT}; return bin_level(unary,o,3); }
static Node *add(void) { const TokType o[]={T_PLUS,T_MINUS}; return bin_level(mul,o,2); }
static Node *compare(void) { const TokType o[]={T_LT,T_LE,T_GT,T_GE}; return bin_level(add,o,4); }
static Node *equal(void) { const TokType o[]={T_EQ,T_NE}; return bin_level(compare,o,2); }
static Node *logical_and(void) { const TokType o[]={T_AND}; return bin_level(equal,o,1); }
static Node *logical_or(void) { const TokType o[]={T_OR}; return bin_level(logical_and,o,1); }
static Node *expression(void) { return logical_or(); }

static Node *block(void) {
    Token t=expect(T_LB,"代码块需要 '{'"); Node*n=node(N_BLOCK,t);
    while(cur()->type!=T_RB && cur()->type!=T_EOF) {
        if(n->count>=128) fatal_at(*cur(),"单个代码块语句过多（上限 128）");
        n->items[n->count++]=statement();
    }
    expect(T_RB,"代码块缺少 '}'"); return n;
}
static Node *statement(void) {
    Token t=*cur();
    if(match(T_LB)) { pos--; return block(); }
    if(match(T_LET)) {
        Node*n=node(N_LET,t); Token id=expect(T_ID,"let 后需要变量名");
        snprintf(n->text,sizeof(n->text),"%s",id.text);
        expect(T_ASSIGN,"变量定义需要 '='"); n->a=expression(); expect(T_SEMI,"语句末尾需要 ';'"); return n;
    }
    if(match(T_IF)) {
        Node*n=node(N_IF,t); expect(T_LP,"if 后需要 '('"); n->a=expression();
        expect(T_RP,"if 条件后需要 ')'"); n->b=statement();
        if(match(T_ELSE)) n->c=statement();
        return n;
    }
    if(match(T_FOR)) {
        Node*n=node(N_FOR,t); expect(T_LP,"for 后需要 '('");
        if(match(T_SEMI)) n->a=NULL;
        else if(match(T_LET)) {
            Node*x=node(N_LET,t); Token id=expect(T_ID,"for 初始化需要变量名");
            snprintf(x->text,sizeof(x->text),"%s",id.text); expect(T_ASSIGN,"变量定义需要 '='");
            x->a=expression(); expect(T_SEMI,"for 初始化后需要 ';'"); n->a=x;
        } else {
            Node*x=node(N_ASSIGN,*cur()); Token id=expect(T_ID,"for 初始化需要变量名");
            snprintf(x->text,sizeof(x->text),"%s",id.text); expect(T_ASSIGN,"赋值需要 '='");
            x->a=expression(); expect(T_SEMI,"for 初始化后需要 ';'"); n->a=x;
        }
        if(cur()->type!=T_SEMI) n->b=expression();
        expect(T_SEMI,"for 条件后需要 ';'");
        if(cur()->type!=T_RP) {
            Token id=expect(T_ID,"for 更新部分需要变量赋值");
            Node*x=node(N_ASSIGN,id); snprintf(x->text,sizeof(x->text),"%s",id.text);
            expect(T_ASSIGN,"for 更新部分需要 '='"); x->a=expression(); n->c=x;
        }
        expect(T_RP,"for 头部需要 ')'"); if(n->count>=128) fatal_at(t,"内部错误");
        n->items[n->count++]=statement(); return n;
    }
    if(match(T_FN)) {
        Node*n=node(N_FN,t); Token id=expect(T_ID,"fn 后需要函数名");
        snprintf(n->text,sizeof(n->text),"%s",id.text); expect(T_LP,"函数参数列表需要 '('");
        if(cur()->type!=T_RP) do {
            if(n->nparams>=MAX_PARAMS) fatal_at(*cur(),"函数最多支持 %d 个参数",MAX_PARAMS);
            Token p=expect(T_ID,"参数必须是变量名"); n->params[n->nparams++]=dupstr(p.text);
        } while(match(T_COMMA));
        expect(T_RP,"参数列表需要 ')'"); n->a=block(); return n;
    }
    if(match(T_RETURN)) {
        Node*n=node(N_RETURN,t); if(cur()->type!=T_SEMI) n->a=expression();
        expect(T_SEMI,"return 语句末尾需要 ';'"); return n;
    }
    if(cur()->type==T_ID && toks[pos+1].type==T_ASSIGN) {
        Token id=*cur(); pos+=2; Node*n=node(N_ASSIGN,id); snprintf(n->text,sizeof(n->text),"%s",id.text);
        n->a=expression(); expect(T_SEMI,"赋值语句末尾需要 ';'"); return n;
    }
    Node*n=node(N_EXPR,t); n->a=expression(); expect(T_SEMI,"语句末尾需要 ';'"); return n;
}

typedef struct { char name[128]; double value; } Var;
typedef struct Env { Var vars[MAX_VARS]; int count; struct Env *parent; } Env;
typedef struct { char name[128]; Node *decl; } Func;
static Func funcs[MAX_FUNCS]; static int nfuncs;
static int call_depth, loop_steps;
typedef struct { double value; int is_string; char string[MAX_TEXT]; } Value;
typedef struct { int returned; Value value; } Flow;

static int find_var(Env *e,const char *name) {
    for(int i=e->count-1;i>=0;i--) if(!strcmp(e->vars[i].name,name)) return i;
    return -1;
}
static int has_var(Env *e,const char *name) { return find_var(e,name)>=0; }
static double get_var(Env *e,Token t,const char *name) {
    for(Env *p=e;p;p=p->parent) { int i=find_var(p,name); if(i>=0) return p->vars[i].value; }
    fatal_at(t,"未定义变量 '%s'",name); return 0;
}
static void define_var(Env *e,Token t,const char *name,double v) {
    if(has_var(e,name)) fatal_at(t,"变量 '%s' 在当前作用域中已定义",name);
    if(e->count>=MAX_VARS) fatal_at(t,"变量数量超出上限");
    snprintf(e->vars[e->count].name,sizeof(e->vars[e->count].name),"%s",name);
    e->vars[e->count++].value=v;
}
static void set_var(Env *e,Token t,const char *name,double v) {
    for(Env *p=e;p;p=p->parent) { int i=find_var(p,name); if(i>=0) { p->vars[i].value=v; return; } }
    fatal_at(t,"不能给未定义变量 '%s' 赋值",name);
}
static Value number(double x) { Value v={0}; v.value=x; return v; }
static Value eval(Node *n,Env *e);
static Flow exec(Node *n,Env *e);
static Value call_builtin(Node *n,Env *e,int *handled) {
    *handled=1; int c=n->count; double a=0,b=0;
    #define ARITY(lo,hi) do { if(c<(lo)||c>(hi)) fatal_at(n->tok,"函数 '%s' 参数数量错误：需要 %d 到 %d 个，实际 %d 个",n->text,(lo),(hi),c); } while(0)
    if(!strcmp(n->text,"print")) {
        ARITY(1,1);
        Node *arg=n->items[0];
        if(arg->type!=N_STR) { Value v=eval(arg,e); if(v.is_string) fatal_at(n->tok,"print 参数必须是数字或字符串模板"); printf("%.15g",v.value); return number(v.value); }
        const char *s=arg->text;
        while(*s) {
            if(s[0]=='{' ) {
                const char *end=strchr(s+1,'}');
                if(end) {
                    size_t len=(size_t)(end-(s+1)); char name[128];
                    if(len==0 || len>=sizeof(name)) fatal_at(arg->tok,"模板变量名无效");
                    memcpy(name,s+1,len); name[len]=0;
                    double v=get_var(e,arg->tok,name); printf("%.15g",v); s=end+1; continue;
                }
            }
            putchar(*s++);
        }
        return number(0);
    }
    if(c>0) { Value v=eval(n->items[0],e); if(v.is_string) fatal_at(n->tok,"函数 '%s' 的参数必须是数字",n->text); a=v.value; }
    if(c>1) { Value v=eval(n->items[1],e); if(v.is_string) fatal_at(n->tok,"函数 '%s' 的参数必须是数字",n->text); b=v.value; }
    if(!strcmp(n->text,"abs")) { ARITY(1,1); return number(fabs(a)); }
    if(!strcmp(n->text,"round")) { ARITY(1,1); return number(floor(a+0.5)); }
    if(!strcmp(n->text,"sqrt")) { ARITY(1,1); if(a<0) fatal_at(n->tok,"sqrt 参数不能为负数"); return number(sqrt(a)); }
    if(!strcmp(n->text,"floor")) { ARITY(1,1); return number(floor(a)); }
    if(!strcmp(n->text,"ceil")) { ARITY(1,1); return number(ceil(a)); }
    if(!strcmp(n->text,"pow")) { ARITY(2,2); return number(pow(a,b)); }
    if(!strcmp(n->text,"min")) { ARITY(2,2); return number(a<b?a:b); }
    if(!strcmp(n->text,"max")) { ARITY(2,2); return number(a>b?a:b); }
    *handled=0; return number(0);
    #undef ARITY
}
static Value eval(Node *n,Env *e) {
    if(!n) return number(0);
    switch(n->type) {
        case N_NUM: return number(n->num);
        case N_STR: { Value v=number(0); v.is_string=1; snprintf(v.string,sizeof(v.string),"%s",n->text); return v; }
        case N_VAR: return number(get_var(e,n->tok,n->text));
        case N_UNARY: {
            Value a=eval(n->a,e); if(a.is_string) fatal_at(n->tok,"一元运算符不能用于字符串");
            if(n->op==T_MINUS) return number(-a.value);
            if(n->op==T_PLUS) return number(a.value);
            return number(!a.value);
        }
        case N_BINARY: {
            Value x=eval(n->a,e);
            if(n->op==T_AND && !x.value) return number(0);
            if(n->op==T_OR && x.value) return number(1);
            Value y=eval(n->b,e);
            if(x.is_string||y.is_string) fatal_at(n->tok,"算术和比较运算只能用于数字");
            double a=x.value,b=y.value;
            switch(n->op) {
                case T_PLUS:return number(a+b); case T_MINUS:return number(a-b);
                case T_STAR:return number(a*b);
                case T_SLASH:if(b==0) fatal_at(n->tok,"除数不能为 0"); return number(a/b);
                case T_PERCENT:if(b==0) fatal_at(n->tok,"取余运算的除数不能为 0"); return number(fmod(a,b));
                case T_EQ:return number(a==b); case T_NE:return number(a!=b);
                case T_LT:return number(a<b); case T_LE:return number(a<=b);
                case T_GT:return number(a>b); case T_GE:return number(a>=b);
                case T_AND:return number(a!=0 && b!=0); case T_OR:return number(a!=0 || b!=0);
            }
            break;
        }
        case N_CALL: {
            int handled=0; Value v=call_builtin(n,e,&handled); if(handled) return v;
            Func *f=NULL; for(int i=0;i<nfuncs;i++) if(!strcmp(funcs[i].name,n->text)) { f=&funcs[i]; break; }
            if(!f) fatal_at(n->tok,"未知函数 '%s'",n->text);
            Node *d=f->decl;
            if(n->count!=d->nparams) fatal_at(n->tok,"函数 '%s' 需要 %d 个参数，实际传入 %d 个",n->text,d->nparams,n->count);
            if(call_depth>=MAX_CALL_DEPTH) fatal_at(n->tok,"函数递归深度超过 %d",MAX_CALL_DEPTH);
            double args[MAX_PARAMS];
            for(int i=0;i<n->count;i++) { Value v=eval(n->items[i],e); if(v.is_string) fatal_at(n->tok,"函数参数必须是数字"); args[i]=v.value; }
            Env local={0}; local.parent=e; call_depth++;
            for(int i=0;i<d->nparams;i++) define_var(&local,d->tok,d->params[i],args[i]);
            Flow flow=exec(d->a,&local); call_depth--;
            return flow.returned?flow.value:number(0);
        }
        default: fatal_at(n->tok,"此处不是表达式"); 
    }
    return number(0);
}
static Flow noflow(void) { Flow f={0}; f.value=number(0); return f; }
static Flow exec(Node *n,Env *e) {
    if(!n) return noflow();
    Flow f=noflow();
    switch(n->type) {
        case N_BLOCK: {
            Env local={0}; local.parent=e;
            for(int i=0;i<n->count;i++) { f=exec(n->items[i],&local); if(f.returned) return f; }
            return noflow();
        }
        case N_LET: { Value v=eval(n->a,e); if(v.is_string) fatal_at(n->tok,"变量目前只支持数字类型"); define_var(e,n->tok,n->text,v.value); return f; }
        case N_ASSIGN: { Value v=eval(n->a,e); if(v.is_string) fatal_at(n->tok,"变量目前只支持数字类型"); set_var(e,n->tok,n->text,v.value); return f; }
        case N_EXPR: (void)eval(n->a,e); return f;
        case N_RETURN:
            if(call_depth==0) fatal_at(n->tok,"return 只能在函数内部使用");
            f.returned=1; f.value=n->a?eval(n->a,e):number(0); return f;
        case N_IF: {
            Value v=eval(n->a,e); if(v.is_string) fatal_at(n->tok,"if 条件必须是数字/比较表达式");
            return exec(v.value?n->b:n->c,e);
        }
        case N_FOR: {
            Env local={0}; local.parent=e;
            if(n->a) { f=exec(n->a,&local); if(f.returned) return f; }
            while(!n->b || eval(n->b,&local).value!=0) {
                if(++loop_steps>MAX_LOOP_STEPS) fatal_at(n->tok,"循环执行超过安全上限 %d 步",MAX_LOOP_STEPS);
                f=exec(n->items[0],&local); if(f.returned) return f;
                if(n->c) { f=exec(n->c,&local); if(f.returned) return f; }
            }
            return noflow();
        }
        case N_FN: return f; /* 函数在执行前统一登记 */
        default: fatal_at(n->tok,"无效语句节点");
    }
    return f;
}
static void register_functions(Node *root) {
    if(!root || root->type!=N_BLOCK) return;
    for(int i=0;i<root->count;i++) {
        Node *n=root->items[i];
        if(n->type==N_FN) {
            for(int j=0;j<nfuncs;j++) if(!strcmp(funcs[j].name,n->text))
                fatal_at(n->tok,"函数 '%s' 重复定义",n->text);
            if(nfuncs>=MAX_FUNCS) fatal_at(n->tok,"函数数量超出上限");
            snprintf(funcs[nfuncs].name,sizeof(funcs[nfuncs].name),"%s",n->text);
            funcs[nfuncs++].decl=n;
        }
    }
}
static char *read_file(const char *path) {
    FILE *fp=fopen(path,"rb"); if(!fp) { perror(path); exit(1); }
    if(fseek(fp,0,SEEK_END)!=0) { perror("fseek"); exit(1); }
    long len=ftell(fp); if(len<0 || len>4*1024*1024) { fprintf(stderr,"文件太大或无法读取\n"); exit(1); }
    rewind(fp); char *b=(char*)malloc((size_t)len+1); if(!b) { perror("malloc"); exit(1); }
    size_t got=fread(b,1,(size_t)len,fp); fclose(fp); b[got]=0; return b;
}
int main(int argc,char **argv) {
    if(argc!=2) {
        fprintf(stderr,"用法: %s script.ml\n",argv[0]);
        fprintf(stderr,"示例: fn square(x) { return x * x; } print(\"{square(5)}\\n\");\n");
        return 2;
    }
    src=read_file(argv[1]); lex();
    Node *root=node(N_BLOCK,toks[0]);
    while(cur()->type!=T_EOF) {
        if(root->count>=128) fatal_at(*cur(),"顶层语句过多（上限 128）");
        root->items[root->count++]=statement();
    }
    register_functions(root);
    Env global={0};
    for(int i=0;i<root->count;i++) {
        Node *n=root->items[i];
        if(n->type==N_FN) continue;
        Flow f=exec(n,&global);
        if(f.returned) fatal_at(n->tok,"return 不能出现在函数外部");
    }
    free((void*)src);
    return had_error?1:0;
}

