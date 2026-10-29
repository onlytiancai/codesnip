/*
 * 对照实验版本（不是教程正文的一部分）
 *
 * 这是 ../../minilang.c 的一份改动过的副本：只把 Env 的变量表
 * 从"按需增长的动态数组"改成了"固定长度数组 Var vars[2048]"，
 * 其余代码一个字没动。
 *
 * 目的：让你亲眼看到——如果第 5 步选了固定数组，递归到第 8 层就会段错误。
 * 跑 ./run.sh 看对比结果。
 *
 * MiniLang —— 用 C 实现一门小型解释型语言
 *
 * 编译: cc -std=c11 -Wall -Wextra -O2 minilang.c -lm -o minilang
 * 运行: ./minilang 脚本.ml [tokens|ast|run]
 *
 * 语法速览:
 *   let x = 10;              定义变量
 *   x = x + 1;               赋值
 *   print("x = {x}\n");      打印（模板里只支持 {变量名}）
 *   if (x > 10) { ... } else { ... }
 *   for (let i = 0; i < 5; i = i + 1) { ... }
 *   fn add(a, b) { return a + b; }
 *
 * 三种运行模式:
 *   tokens  只做词法分析，打印 token 列表
 *   ast     词法 + 语法分析，打印抽象语法树
 *   run     完整执行程序（默认）
 *
 * 说明：文件头注释在教程所有步骤中保持完全一致，这样相邻两步的 diff
 * 就只剩下真正的功能改动，读者不会被无关的噪音干扰。
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <stdarg.h>

#define MAX_TOKENS 20000
#define MAX_NODES  20000
#define MAX_TEXT   4096
#define MAX_PARAMS 32
#define MAX_VARS   2048
#define MAX_NAME   128
#define MAX_FUNCS  256
#define MAX_CALL_DEPTH 128
#define MAX_LOOP_STEPS 1000000

/* ------------------------------------------------------------------ */
/* Token：词法分析器产出的最小单位                                     */
/* ------------------------------------------------------------------ */
typedef enum {
    T_EOF, T_NUM, T_ID, T_STR,
    T_LET, T_FN, T_RETURN, T_IF, T_ELSE, T_FOR,
    T_LP, T_RP, T_LB, T_RB, T_COMMA, T_SEMI,
    T_PLUS, T_MINUS, T_STAR, T_SLASH, T_PERCENT,
    T_ASSIGN, T_EQ, T_NE, T_LT, T_LE, T_GT, T_GE,
    T_AND, T_OR, T_NOT
} TokType;

typedef struct {
    TokType type;      /* 这个 token 是什么类别            */
    char    text[MAX_TEXT]; /* 原始文本（标识符/数字/字符串内容） */
    double  num;       /* 数字字面量的数值                 */
    int     line, col; /* 出错时要告诉用户"哪一行哪一列"    */
} Token;

static Token toks[MAX_TOKENS];   /* token 数组 */
static int   ntok;               /* 已经识别了多少个 token */
static int   pos;                /* 语法分析器当前读到第几个 token */
static const char *src;          /* 整份源代码 */
static int   src_line, src_col;  /* 扫描器当前所在的位置 */
static int   had_error;          /* 出过错没有 */
static int   loop_steps;         /* 循环一共执行了多少步（用来防死循环） */

/* ------------------------------------------------------------------ */
/* 错误处理：从第一步就定好名字和签名，后面不再改                       */
/* ------------------------------------------------------------------ */
static void error_at(int line, int col, const char *fmt, ...) {
    va_list ap;
    fprintf(stderr, "错误 [%d:%d]: ", line, col);
    va_start(ap, fmt); vfprintf(stderr, fmt, ap); va_end(ap);
    fputc('\n', stderr);
    had_error = 1;
}

/* 致命错误：打印后直接结束程序（教学版采取最简单的策略） */
static _Noreturn void fatal_at(Token t, const char *fmt, ...) {
    va_list ap;
    fprintf(stderr, "错误 [%d:%d]: ", t.line, t.col);
    va_start(ap, fmt); vfprintf(stderr, fmt, ap); va_end(ap);
    fputc('\n', stderr);
    had_error = 1;
    exit(1);
}

/* 复制一份字符串（参数名要保存到运行期，所以必须复制，不能指着源码） */
static char *dupstr(const char *s) {
    size_t n = strlen(s) + 1;
    char *p = (char *)malloc(n);
    if (!p) { perror("malloc"); exit(1); }
    memcpy(p, s, n);
    return p;
}

/* 标识符可以以字母或下划线开头 */
static int is_id_start(int c) { return isalpha((unsigned char)c) || c == '_'; }
/* 标识符的后续字符可以是字母、数字或下划线 */
static int is_id_char(int c)  { return isalnum((unsigned char)c) || c == '_'; }

/* ------------------------------------------------------------------ */
/* 词法分析器                                                         */
/* ------------------------------------------------------------------ */
static void add_token(TokType type, const char *text, double num, int line, int col) {
    if (ntok >= MAX_TOKENS) { error_at(line, col, "Token 数量超出上限"); exit(1); }
    Token *t = &toks[ntok++];
    t->type = type;
    t->num  = num;
    t->line = line;
    t->col  = col;
    snprintf(t->text, sizeof(t->text), "%s", text ? text : "");
}

/* 把源文件扫描成一串 token */
static void lex(void) {
    ntok = 0; src_line = 1; src_col = 1;
    size_t i = 0;
    while (src[i]) {
        int c = (unsigned char)src[i];

        /* 空白字符：跳过，但列号要跟着走 */
        if (c == ' ' || c == '\t' || c == '\r') { i++; src_col++; continue; }
        if (c == '\n') { i++; src_line++; src_col = 1; continue; }

        /* 行注释：// 后面到行尾全部忽略 */
        if (c == '/' && src[i+1] == '/') {
            while (src[i] && src[i] != '\n') { i++; src_col++; }
            continue;
        }

        int line = src_line, col = src_col;

        /* 数字：整数或小数 */
        if (isdigit(c) || (c == '.' && isdigit((unsigned char)src[i+1]))) {
            size_t start = i;
            int dots = 0;
            while (isdigit((unsigned char)src[i]) || src[i] == '.') {
                if (src[i] == '.' && ++dots > 1) break;   /* 最多一个小数点 */
                i++; src_col++;
            }
            char b[128];
            size_t n = i - start;
            if (n >= sizeof(b)) fatal_at((Token){.line=line, .col=col}, "数字字面量太长");
            memcpy(b, src + start, n); b[n] = 0;
            char *end = NULL;
            double v = strtod(b, &end);
            if (!end || *end) fatal_at((Token){.line=line, .col=col}, "非法数字 '%s'", b);
            add_token(T_NUM, b, v, line, col);
            continue;
        }

        /* 标识符 / 关键字 */
        if (is_id_start(c)) {
            size_t start = i;
            while (is_id_char((unsigned char)src[i])) { i++; src_col++; }
            char b[128];
            size_t n = i - start;
            if (n >= sizeof(b)) fatal_at((Token){.line=line, .col=col}, "标识符太长");
            memcpy(b, src + start, n); b[n] = 0;
            TokType ty = T_ID;
            if      (!strcmp(b, "let"))    ty = T_LET;
            else if (!strcmp(b, "fn"))     ty = T_FN;
            else if (!strcmp(b, "return")) ty = T_RETURN;
            else if (!strcmp(b, "if"))     ty = T_IF;
            else if (!strcmp(b, "else"))   ty = T_ELSE;
            else if (!strcmp(b, "for"))    ty = T_FOR;
            add_token(ty, b, 0, line, col);
            continue;
        }

        /* 字符串字面量 */
        if (c == '"') {
            i++; src_col++;
            char b[MAX_TEXT];
            size_t n = 0;
            int closed = 0;
            while (src[i]) {
                if (src[i] == '"') { i++; src_col++; closed = 1; break; }
                if (src[i] == '\n')
                    fatal_at((Token){.line=line, .col=col}, "字符串不能直接跨行，请使用 \\n");
                char ch = src[i++]; src_col++;
                if (ch == '\\' && src[i]) {         /* 转义序列 */
                    char e = src[i++]; src_col++;
                    if      (e == 'n')  ch = '\n';
                    else if (e == 't')  ch = '\t';
                    else if (e == 'r')  ch = '\r';
                    else if (e == '"' || e == '\\') ch = e;
                    else fatal_at((Token){.line=line, .col=col}, "不支持的字符串转义 \\%c", e);
                }
                if (n + 1 >= sizeof(b)) fatal_at((Token){.line=line, .col=col}, "字符串太长");
                b[n++] = ch;
            }
            if (!closed) fatal_at((Token){.line=line, .col=col}, "字符串缺少结束引号");
            b[n] = 0;
            add_token(T_STR, b, 0, line, col);
            continue;
        }

        /* 运算符：先尝试双字符的（== != <= >= && ||），再退回单字符 */
        TokType ty = T_EOF;
        char two[3] = {0, 0, 0};
        two[0] = (char)c; two[1] = src[i+1];
        if      (!strncmp(two, "==", 2)) ty = T_EQ;
        else if (!strncmp(two, "!=", 2)) ty = T_NE;
        else if (!strncmp(two, "<=", 2)) ty = T_LE;
        else if (!strncmp(two, ">=", 2)) ty = T_GE;
        else if (!strncmp(two, "&&", 2)) ty = T_AND;
        else if (!strncmp(two, "||", 2)) ty = T_OR;
        if (ty != T_EOF) { i += 2; src_col += 2; add_token(ty, two, 0, line, col); continue; }

        switch (c) {
            case '(':  ty = T_LP;      break;
            case ')':  ty = T_RP;      break;
            case '{':  ty = T_LB;      break;
            case '}':  ty = T_RB;      break;
            case ',':  ty = T_COMMA;   break;
            case ';':  ty = T_SEMI;    break;
            case '+':  ty = T_PLUS;    break;
            case '-':  ty = T_MINUS;   break;
            case '*':  ty = T_STAR;    break;
            case '/':  ty = T_SLASH;   break;
            case '%':  ty = T_PERCENT; break;
            case '=':  ty = T_ASSIGN;  break;
            case '<':  ty = T_LT;      break;
            case '>':  ty = T_GT;      break;
            case '!':  ty = T_NOT;    break;
            default:   fatal_at((Token){.line=line, .col=col}, "无法识别字符 '%c'", c);
        }
        i++; src_col++;
        add_token(ty, two, 0, line, col);
    }
    add_token(T_EOF, "<文件结束>", 0, src_line, src_col);
}

/* token 类型的名字，打印时用 */
static const char *type_name(TokType t) {
    switch (t) {
        case T_EOF:     return "EOF";
        case T_NUM:     return "NUM";
        case T_ID:      return "ID";
        case T_STR:     return "STR";
        case T_LET:     return "LET";
        case T_FN:      return "FN";
        case T_RETURN:  return "RETURN";
        case T_IF:      return "IF";
        case T_ELSE:    return "ELSE";
        case T_FOR:     return "FOR";
        case T_LP:      return "LP";
        case T_RP:      return "RP";
        case T_LB:      return "LB";
        case T_RB:      return "RB";
        case T_COMMA:   return "COMMA";
        case T_SEMI:    return "SEMI";
        case T_PLUS:    return "PLUS";
        case T_MINUS:   return "MINUS";
        case T_STAR:    return "STAR";
        case T_SLASH:   return "SLASH";
        case T_PERCENT: return "PERCENT";
        case T_ASSIGN:  return "ASSIGN";
        case T_EQ:      return "EQ";
        case T_NE:      return "NE";
        case T_LT:      return "LT";
        case T_LE:      return "LE";
        case T_GT:      return "GT";
        case T_GE:      return "GE";
        case T_AND:     return "AND";
        case T_OR:      return "OR";
        case T_NOT:     return "NOT";
    }
    return "?";
}

/* ------------------------------------------------------------------ */
/* AST：抽象语法树                                                     */
/* ------------------------------------------------------------------ */
typedef enum {
    N_NUM,      /* 数字字面量        */
    N_STR,      /* 字符串字面量      */
    N_VAR,      /* 变量引用          */
    N_UNARY,    /* 一元运算：-x !x   */
    N_BINARY,   /* 二元运算：a + b   */
    N_CALL,     /* 函数调用：f(a,b)  */
    N_BLOCK,    /* 代码块 { ... }    */
    N_LET,      /* let x = 1;        */
    N_ASSIGN,   /* x = 1;            */
    N_EXPR,     /* 表达式语句        */
    N_IF,       /* if (条件) ... else ... */
    N_FOR,      /* for (init; cond; update) { ... } */
    N_FN,       /* fn 名字(参数) { ... } */
    N_RETURN    /* return 表达式; */
} NodeType;

typedef struct Node Node;
struct Node {
    NodeType type;
    Token   tok;              /* 出错定位用：行列号 */
    double  num;              /* N_NUM 的数值      */
    char    text[MAX_TEXT];   /* N_VAR/N_CALL 的名字，N_STR 的内容 */
    int     op;               /* 运算符的 TokType  */
    Node   *a, *b, *c;        /* 子节点            */
    Node   *items[128];       /* N_CALL 的实参列表 */
    int     count;
    char   *params[MAX_PARAMS]; /* 函数参数名（后面几步才用） */
    int     nparams;
};

/* 节点池：所有 AST 节点都从这里分配，用完为止 */
static Node nodes[MAX_NODES];
static int  nnodes;

/* 创建一个节点。memset 清零非常重要：否则 switch 里没覆盖到的字段
   会是随机值，后面调试会非常痛苦 */
static Node *node(NodeType ty, Token t) {
    if (nnodes >= MAX_NODES) fatal_at(t, "语法树节点超出上限");
    Node *n = &nodes[nnodes++];
    memset(n, 0, sizeof(*n));
    n->type = ty;
    n->tok  = t;
    return n;
}

/* ------------------------------------------------------------------ */
/* 语法分析器：递归下降                                                 */
/* ------------------------------------------------------------------ */
static Node *expression(void);

static Token *cur(void) { return &toks[pos]; }

/* 当前 token 是不是 t？是就吃掉它，返回 1；不是就什么都不做，返回 0 */
static int match(TokType t) {
    if (cur()->type == t) { pos++; return 1; }
    return 0;
}

/* 当前 token 必须是 t，否则报错并退出 */
static Token expect(TokType t, const char *what) {
    if (cur()->type != t)
        fatal_at(*cur(), "语法错误：%s，当前是 '%s'", what, cur()->text);
    return toks[pos++];
}

/* 最基础的东西：数字、字符串、变量、函数调用、括号表达式 */
static Node *primary(void) {
    Token t = *cur();

    if (match(T_NUM)) {
        Node *n = node(N_NUM, t);
        n->num = t.num;
        return n;
    }
    if (match(T_STR)) {
        Node *n = node(N_STR, t);
        snprintf(n->text, sizeof(n->text), "%s", t.text);
        return n;
    }
    if (match(T_ID)) {
        /* 标识符后面紧跟 '('，说明是函数调用 */
        if (match(T_LP)) {
            Node *n = node(N_CALL, t);
            snprintf(n->text, sizeof(n->text), "%s", t.text);
            if (cur()->type != T_RP) {
                do {
                    if (n->count >= 128) fatal_at(*cur(), "函数参数过多");
                    n->items[n->count++] = expression();
                } while (match(T_COMMA));
            }
            expect(T_RP, "函数调用需要 ')'");
            return n;
        }
        Node *n = node(N_VAR, t);
        snprintf(n->text, sizeof(n->text), "%s", t.text);
        return n;
    }
    if (match(T_LP)) {
        Node *n = expression();
        expect(T_RP, "表达式需要 ')'");
        return n;
    }

    fatal_at(t, "语法错误：此处需要数字、变量、函数调用或括号表达式");
    return NULL;
}

/* 一元运算：-x +x !x */
static Node *unary(void) {
    if (cur()->type == T_MINUS || cur()->type == T_PLUS || cur()->type == T_NOT) {
        Token t = *cur();
        pos++;
        Node *n = node(N_UNARY, t);
        n->op = t.type;
        n->a  = unary();
        return n;
    }
    return primary();
}

/* 一层二元运算：左结合的循环 */
static Node *bin_level(Node *(*next)(void), const TokType *ops, int nops) {
    Node *left = next();
    for (;;) {
        int found = 0;
        for (int i = 0; i < nops; i++) {
            if (cur()->type == ops[i]) {
                Token t = *cur();
                pos++;
                Node *n = node(N_BINARY, t);
                n->op = t.type;
                n->a  = left;
                n->b  = next();
                left  = n;
                found = 1;
                break;
            }
        }
        if (!found) return left;
    }
}

static Node *mul(void) {           /* * / % */
    const TokType o[] = {T_STAR, T_SLASH, T_PERCENT};
    return bin_level(unary, o, 3);
}
static Node *add(void) {           /* + - */
    const TokType o[] = {T_PLUS, T_MINUS};
    return bin_level(mul, o, 2);
}
static Node *compare(void) {       /* < <= > >= */
    const TokType o[] = {T_LT, T_LE, T_GT, T_GE};
    return bin_level(add, o, 4);
}
static Node *equal(void) {         /* == != */
    const TokType o[] = {T_EQ, T_NE};
    return bin_level(compare, o, 2);
}
static Node *logical_and(void) {   /* && */
    const TokType o[] = {T_AND};
    return bin_level(equal, o, 1);
}
static Node *logical_or(void) {    /* || */
    const TokType o[] = {T_OR};
    return bin_level(logical_and, o, 1);
}

/* 表达式的最外层入口 */
static Node *expression(void) { return logical_or(); }

/* ------------------------------------------------------------------ */
/* 语法分析器：语句                                                    */
/* ------------------------------------------------------------------ */
static Node *statement(void);

/* 代码块：{ 语句 语句 ... } */
static Node *block(void) {
    Token t = expect(T_LB, "代码块需要 '{'");
    Node *n = node(N_BLOCK, t);
    while (cur()->type != T_RB && cur()->type != T_EOF) {
        if (n->count >= 128) fatal_at(*cur(), "单个代码块语句过多（上限 128）");
        n->items[n->count++] = statement();
    }
    expect(T_RB, "代码块缺少 '}'");
    return n;
}

/* 一条语句 */
static Node *statement(void) {
    Token t = *cur();

    /* 裸代码块 { ... } */
    if (cur()->type == T_LB) return block();

    /* return 表达式; */
    if (match(T_RETURN)) {
        Node *n = node(N_RETURN, t);
        if (cur()->type != T_SEMI) n->a = expression();   /* return; 也可以不带值 */
        expect(T_SEMI, "return 语句末尾需要 ';'");
        return n;
    }

    /* 函数定义：fn 名字(参数, ...) { ... } */
    if (match(T_FN)) {
        Node *n = node(N_FN, t);
        Token id = expect(T_ID, "fn 后需要函数名");
        snprintf(n->text, sizeof(n->text), "%s", id.text);
        expect(T_LP, "函数参数列表需要 '('");
        if (cur()->type != T_RP) {
            do {
                if (n->nparams >= MAX_PARAMS)
                    fatal_at(*cur(), "函数最多支持 %d 个参数", MAX_PARAMS);
                Token p = expect(T_ID, "参数必须是变量名");
                n->params[n->nparams++] = dupstr(p.text);   /* 参数名要留到运行时用，复制一份 */
            } while (match(T_COMMA));
        }
        expect(T_RP, "参数列表需要 ')'");
        n->a = block();      /* 函数体也是一个代码块 */
        return n;
    }

    /* if (条件) 语句 [else 语句] */
    if (match(T_IF)) {
        Node *n = node(N_IF, t);
        expect(T_LP, "if 后需要 '('");
        n->a = expression();
        expect(T_RP, "if 条件后需要 ')'");
        n->b = statement();          /* 条件为真时执行的分支 */
        if (match(T_ELSE)) n->c = statement();  /* 条件为假时执行的分支（可以没有） */
        return n;
    }

    /* for (初始化; 条件; 更新) 语句
       三段之间用分号隔开，整个 for 语句本身不需要分号结尾。 */
    if (match(T_FOR)) {
        Node *n = node(N_FOR, t);
        expect(T_LP, "for 后需要 '('");

        /* 第一段：初始化。可以是 let i = 0; 也可以是 i = 0; 也可以留空 */
        if (match(T_SEMI)) {
            n->a = NULL;
        } else if (match(T_LET)) {
            Node *x = node(N_LET, t);
            Token id = expect(T_ID, "for 初始化需要变量名");
            snprintf(x->text, sizeof(x->text), "%s", id.text);
            expect(T_ASSIGN, "变量定义需要 '='");
            x->a = expression();
            expect(T_SEMI, "for 初始化后需要 ';'");
            n->a = x;
        } else {
            Node *x = node(N_ASSIGN, *cur());
            Token id = expect(T_ID, "for 初始化需要变量名");
            snprintf(x->text, sizeof(x->text), "%s", id.text);
            expect(T_ASSIGN, "赋值需要 '='");
            x->a = expression();
            expect(T_SEMI, "for 初始化后需要 ';'");
            n->a = x;
        }

        /* 第二段：条件。留空表示永远为真 */
        if (cur()->type != T_SEMI) n->b = expression();
        expect(T_SEMI, "for 条件后需要 ';'");

        /* 第三段：更新。目前只支持 i = 表达式 这样的赋值 */
        if (cur()->type != T_RP) {
            Token id = expect(T_ID, "for 更新部分需要变量赋值");
            Node *x = node(N_ASSIGN, id);
            snprintf(x->text, sizeof(x->text), "%s", id.text);
            expect(T_ASSIGN, "for 更新部分需要 '='");
            x->a = expression();
            n->c = x;
        }

        expect(T_RP, "for 头部需要 ')'");
        n->items[n->count++] = statement();   /* 循环体 */
        return n;
    }

    /* let 变量 = 表达式; */
    if (match(T_LET)) {
        Node *n = node(N_LET, t);
        Token id = expect(T_ID, "let 后需要变量名");
        snprintf(n->text, sizeof(n->text), "%s", id.text);
        expect(T_ASSIGN, "变量定义需要 '='");
        n->a = expression();
        expect(T_SEMI, "语句末尾需要 ';'");
        return n;
    }

    /* 赋值语句：标识符 = 表达式;
       这里要特别小心不要把 == 当成 =：
       靠"当前是标识符，且下一个 token 是 T_ASSIGN"来判断。 */
    if (cur()->type == T_ID && toks[pos + 1].type == T_ASSIGN) {
        Token id = *cur();
        pos += 2;
        Node *n = node(N_ASSIGN, id);
        snprintf(n->text, sizeof(n->text), "%s", id.text);
        n->a = expression();
        expect(T_SEMI, "赋值语句末尾需要 ';'");
        return n;
    }

    /* 其它都是表达式语句：表达式; */
    Node *n = node(N_EXPR, t);
    n->a = expression();
    expect(T_SEMI, "语句末尾需要 ';'");
    return n;
}

/* 整个程序：一串语句 */
static Node *parse_program(void) {
    Node *n = node(N_BLOCK, toks[0]);
    while (cur()->type != T_EOF) {
        if (n->count >= 128) fatal_at(*cur(), "顶层语句过多（上限 128）");
        n->items[n->count++] = statement();
    }
    return n;
}

/* ------------------------------------------------------------------ */
/* 打印语法树                                                          */
/* ------------------------------------------------------------------ */
/* 把一棵子树打印成带连接线的树。
   prefix 是"给子节点用的缩进前缀"，逐层往下加 "│   " 或 "    " */
static void dump_ast(Node *n, const char *prefix) {
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

/* ------------------------------------------------------------------ */
/* 内置函数表                                                          */
/*                                                                     */
/* 表里放的是"数字进、数字出"的纯函数。print 不在表里——它要输出，   */
/* 而且下一步还要访问作用域去查模板里的变量，需求不一样。             */
/* ------------------------------------------------------------------ */
typedef double (*BuiltinFn)(double *args, Token tok);

typedef struct {
    const char *name;    /* 函数名         */
    int         min_args;/* 最少几个参数   */
    int         max_args;/* 最多几个参数   */
    BuiltinFn   fn;      /* 具体怎么算     */
} Builtin;

static double bi_abs(double *a, Token tok)   { (void)tok; return fabs(a[0]); }
static double bi_round(double *a, Token tok) { (void)tok; return floor(a[0] + 0.5); }
static double bi_sqrt(double *a, Token tok)  { if (a[0] < 0) fatal_at(tok, "sqrt 参数不能为负数"); return sqrt(a[0]); }
static double bi_floor(double *a, Token tok) { (void)tok; return floor(a[0]); }
static double bi_ceil(double *a, Token tok)  { (void)tok; return ceil(a[0]); }
static double bi_pow(double *a, Token tok)   { (void)tok; return pow(a[0], a[1]); }
static double bi_min(double *a, Token tok)   { (void)tok; return a[0] < a[1] ? a[0] : a[1]; }
static double bi_max(double *a, Token tok)   { (void)tok; return a[0] > a[1] ? a[0] : a[1]; }

static const Builtin builtins[] = {
    { "abs",   1, 1, bi_abs   },
    { "round", 1, 1, bi_round },
    { "sqrt",  1, 1, bi_sqrt  },
    { "floor", 1, 1, bi_floor },
    { "ceil",  1, 1, bi_ceil  },
    { "pow",   2, 2, bi_pow   },
    { "min",   2, 2, bi_min   },
    { "max",   2, 2, bi_max   },
};
static const int nbuiltins = (int)(sizeof(builtins) / sizeof(builtins[0]));

/* 按名字查内置函数 */
static const Builtin *find_builtin(const char *name) {
    for (int i = 0; i < nbuiltins; i++)
        if (!strcmp(builtins[i].name, name)) return &builtins[i];
    return NULL;
}

/* ------------------------------------------------------------------ */
/* 用户自定义函数                                                      */
/* ------------------------------------------------------------------ */
typedef struct {
    char  name[MAX_NAME];  /* 函数名     */
    Node *decl;            /* 函数声明节点（名字、参数、函数体都在里面） */
} Func;

static Func funcs[MAX_FUNCS];
static int  nfuncs;
static int  call_depth;   /* 当前调用了多少层，用来防止无限递归 */

static Func *find_func(const char *name) {
    for (int i = 0; i < nfuncs; i++)
        if (!strcmp(funcs[i].name, name)) return &funcs[i];
    return NULL;
}

/* 登记程序里定义的所有函数。
   必须在执行任何语句之前做完，这样函数之间可以互相调用、也可以递归。 */
static void register_functions(Node *root) {
    for (int i = 0; i < root->count; i++) {
        Node *n = root->items[i];
        if (n->type != N_FN) continue;
        if (find_func(n->text))
            fatal_at(n->tok, "函数 '%s' 重复定义", n->text);
        if (nfuncs >= MAX_FUNCS)
            fatal_at(n->tok, "函数数量超出上限");
        snprintf(funcs[nfuncs].name, sizeof(funcs[nfuncs].name), "%s", n->text);
        funcs[nfuncs].decl = n;
        nfuncs++;
    }
}

/* ------------------------------------------------------------------ */
/* 运行时：值与求值                                                    */
/* ------------------------------------------------------------------ */

/* 一个变量 */
typedef struct {
    char   name[MAX_NAME];
    double value;
} Var;

/* 环境（作用域）：一串变量 + 指向外层作用域的指针。
   第 3 步只是留了个空壳，现在把它真正填上。 */
struct Env {
    Var    vars[2048];  /* 对照实验：固定长度数组！ */
    int    count;
    int    cap;
    struct Env *parent;
};
typedef struct Env Env;   /* 后面就可以直接写 Env 而不是 struct Env */

/* 作用域的建立与销毁 */
static void env_init(Env *e, Env *parent) {
    /* 数组不需要置空 */
    e->count  = 0;
    e->cap    = 0;
    e->parent = parent;
}

static void env_free(Env *e) {   /* 对照实验：数组不需要 free */
    e->count = 0;
    e->cap = 0;
}


/* 在当前作用域里加一个变量。变量表是动态数组：
   用多少占多少，不用一开始就开 2048 个槽位。 */
static void env_push(Env *e, const char *name, double v) {
    if (e->count >= MAX_VARS)
        fatal_at((Token){.line = 0, .col = 0}, "变量数量超出上限");
    e->cap = 2048;   /* 对照实验：容量固定 */
    snprintf(e->vars[e->count].name, sizeof(e->vars[e->count].name), "%s", name);
    e->vars[e->count].value = v;
    e->count++;
}

/* 在当前作用域里找变量。倒着找：后定义的遮蔽先定义的。 */
static int find_var(Env *e, const char *name) {
    for (int i = e->count - 1; i >= 0; i--)
        if (!strcmp(e->vars[i].name, name)) return i;
    return -1;
}

/* 沿着作用域链往外找。找不到就是"未定义变量"。 */
static double get_var(Env *e, Token t, const char *name) {
    for (Env *p = e; p; p = p->parent) {
        int i = find_var(p, name);
        if (i >= 0) return p->vars[i].value;
    }
    fatal_at(t, "未定义变量 '%s'", name);
    return 0;
}

/* 定义变量：只能在当前作用域，同名就是重复定义 */
static void define_var(Env *e, Token t, const char *name, double v) {
    if (find_var(e, name) >= 0)
        fatal_at(t, "变量 '%s' 在当前作用域中已定义", name);
    env_push(e, name, v);
}

/* 赋值：可以改外层作用域的变量 */
static void set_var(Env *e, Token t, const char *name, double v) {
    for (Env *p = e; p; p = p->parent) {
        int i = find_var(p, name);
        if (i >= 0) { p->vars[i].value = v; return; }
    }
    fatal_at(t, "不能给未定义变量 '%s' 赋值", name);
}

/* 打印字符串模板：把 {变量名} 换成变量的值。
   逐字符扫描，遇到 '{' 就找配对的 '}'。 */
static void print_template(const char *s, Env *e, Token tok) {
    while (*s) {
        if (s[0] == '{') {
            const char *end = strchr(s + 1, '}');
            if (end) {
                size_t len = (size_t)(end - (s + 1));
                char name[MAX_NAME];
                if (len == 0 || len >= sizeof(name))
                    fatal_at(tok, "模板里的变量名无效");
                memcpy(name, s + 1, len);
                name[len] = '\0';
                /* 模板里的变量必须已经定义过 */
                double v = get_var(e, tok, name);
                printf("%.15g", v);
                s = end + 1;
                continue;
            }
        }
        putchar(*s++);
    }
}

/* 一个值。MiniLang 里普通变量只能是数字，字符串只出现在 print 里，
   所以 Value 用一个标记位区分两者。
   is_string 和 string 这两个字段现在还没用上，是提前留好的。 */
typedef struct {
    double value;             /* 数值            */
    int    is_string;         /* 1 表示是字符串  */
    char   string[MAX_TEXT];  /* 字符串内容      */
} Value;

/* 造一个数字值 */
static Value number(double x) {
    Value v;
    v.value = x;
    v.is_string = 0;
    v.string[0] = '\0';
    return v;
}

/* 控制流信号：一条语句执行完可能什么都不返回，也可能提前"return"。
   returned 字段从现在起就定型了，后面加 return 语句时不用改 exec 的签名。 */
typedef struct {
    int   returned;  /* 1 表示这条语句链里有 return  */
    Value value;     /* return 带回来的值          */
} Flow;

/* 什么都不返回 */
static Flow noflow(void) {
    Flow f;
    f.returned = 0;
    f.value = number(0);
    return f;
}

/* exec 定义在 eval 后面，但 eval 里要调用它：
   C 语言要求"先声明后使用"，所以这里先写一个函数原型（也叫前向声明）。 */
static Flow exec(Node *n, Env *e);

/* 求值：把一棵表达式树算成一个值。
   递归的结构和语法分析器正好相反：分析时"从左到右展开"，
   求值时"从叶子往回算"。 */
static Value eval(Node *n, Env *e) {
    if (!n) return number(0);

    switch (n->type) {
        case N_NUM:
            return number(n->num);

        case N_STR: {
            /* 字符串字面量：求值结果是一个字符串值 */
            Value v;
            v.value = 0;
            v.is_string = 1;
            snprintf(v.string, sizeof(v.string), "%s", n->text);
            return v;
        }

        case N_VAR:
            /* 变量引用：沿着作用域链找 */
            return number(get_var(e, n->tok, n->text));

        case N_CALL: {
            /* print 特殊处理：它要输出，而且还要访问作用域查模板里的变量 */
            if (!strcmp(n->text, "print")) {
                if (n->count != 1)
                    fatal_at(n->tok, "函数 'print' 参数数量错误：需要 1 个，实际 %d 个", n->count);
                Value v = eval(n->items[0], e);
                if (v.is_string) {
                    /* 字符串参数：当成模板渲染 */
                    print_template(v.string, e, n->tok);
                } else {
                    printf("%.15g", v.value);   /* 数字之间不会自动换行 */
                }
                return number(0);
            }

            /* 其余内建函数：查表 */
            const Builtin *b = find_builtin(n->text);
            if (b) {
                if (n->count < b->min_args || n->count > b->max_args)
                    fatal_at(n->tok, "函数 '%s' 参数数量错误：需要 %d 个，实际 %d 个",
                             b->name, b->min_args, n->count);
                double args[MAX_PARAMS];
                for (int i = 0; i < n->count; i++) {
                    Value v = eval(n->items[i], e);
                    if (v.is_string)
                        fatal_at(n->tok, "函数 '%s' 的参数必须是数字", b->name);
                    args[i] = v.value;
                }
                return number(b->fn(args, n->tok));
            }

            /* 用户定义的函数 */
            Func *f = find_func(n->text);
            if (!f)
                fatal_at(n->tok, "未知函数 '%s'", n->text);
            Node *d = f->decl;
            if (n->count != d->nparams)
                fatal_at(n->tok, "函数 '%s' 需要 %d 个参数，实际传入 %d 个",
                         n->text, d->nparams, n->count);
            if (call_depth >= MAX_CALL_DEPTH)
                fatal_at(n->tok, "函数递归深度超过 %d", MAX_CALL_DEPTH);

            double args[MAX_PARAMS];
            for (int i = 0; i < n->count; i++) {
                Value v = eval(n->items[i], e);
                if (v.is_string)
                    fatal_at(n->tok, "函数参数必须是数字");
                args[i] = v.value;
            }

            /* 每次调用都开一个新的作用域，参数就是里面的变量 */
            Env local;
            env_init(&local, e);
            for (int i = 0; i < d->nparams; i++)
                env_push(&local, d->params[i], args[i]);

            call_depth++;
            Flow flow = exec(d->a, &local);
            call_depth--;
            env_free(&local);

            /* 还没有 return，所以函数的值暂时是 0（下一步会补上） */
            return flow.returned ? flow.value : number(0);
        }

        case N_UNARY: {
            Value a = eval(n->a, e);
            if (a.is_string) fatal_at(n->tok, "一元运算符不能用于字符串");
            if (n->op == T_MINUS) return number(-a.value);
            if (n->op == T_PLUS)  return number(a.value);
            /* ! 的真值规则：0 是假，非 0 是真 */
            return number(a.value == 0);
        }

        case N_BINARY: {
            Value x = eval(n->a, e);

            /* 短路求值：&& 和 || 的右边可能根本不执行。
               0 && (1/0) 不应该报"除数为 0"，这就是短路的意义。 */
            if (n->op == T_AND && x.value == 0) return number(0);
            if (n->op == T_OR  && x.value != 0) return number(1);

            Value y = eval(n->b, e);
            /* 字符串只能出现在 print 里，不能参与运算 */
            if (x.is_string || y.is_string)
                fatal_at(n->tok, "算术和比较运算只能用于数字");
            double a = x.value, b = y.value;
            switch (n->op) {
                case T_PLUS:    return number(a + b);
                case T_MINUS:   return number(a - b);
                case T_STAR:    return number(a * b);
                case T_SLASH:
                    if (b == 0) fatal_at(n->tok, "除数不能为 0");
                    return number(a / b);
                case T_PERCENT:
                    if (b == 0) fatal_at(n->tok, "取余运算的除数不能为 0");
                    return number(fmod(a, b));   /* C 的 % 只支持整数，浮点取余要用 fmod */

                /* 比较运算：结果只有 0（假）和 1（真） */
                case T_EQ: return number(a == b);
                case T_NE: return number(a != b);
                case T_LT: return number(a <  b);
                case T_LE: return number(a <= b);
                case T_GT: return number(a >  b);
                case T_GE: return number(a >= b);

                /* 逻辑运算：两边都是真才算真 */
                case T_AND: return number(a != 0 && b != 0);
                case T_OR:  return number(a != 0 || b != 0);
            }
            break;
        }

        default:
            /* 语句节点不能出现在表达式里 */
            fatal_at(n->tok, "此处不是表达式");
    }
    return number(0);
}

/* 执行一条语句。返回值是一个 Flow，告诉调用者"要不要带着结果往上传"。 */
static Flow exec(Node *n, Env *e) {
    if (!n) return noflow();

    Flow f = noflow();
    switch (n->type) {
        case N_BLOCK: {
            /* 代码块自带一个作用域：里面的 let 不影响外面 */
            Env local;
            env_init(&local, e);
            for (int i = 0; i < n->count; i++) {
                f = exec(n->items[i], &local);
                if (f.returned) { env_free(&local); return f; }
            }
            env_free(&local);
            return noflow();
        }

        case N_LET: {
            Value v = eval(n->a, e);
            if (v.is_string) fatal_at(n->tok, "变量目前只支持数字类型");
            define_var(e, n->tok, n->text, v.value);
            return f;
        }

        case N_ASSIGN: {
            Value v = eval(n->a, e);
            if (v.is_string) fatal_at(n->tok, "变量目前只支持数字类型");
            set_var(e, n->tok, n->text, v.value);
            return f;
        }

        case N_EXPR:
            (void)eval(n->a, e);
            return f;

        case N_IF: {
            /* 条件只求值一次：0 是假，非 0 是真 */
            Value v = eval(n->a, e);
            if (v.is_string) fatal_at(n->tok, "if 的条件必须是数字或比较表达式");
            return exec(v.value ? n->b : n->c, e);
            /* 没有 else 时 n->c 是 NULL，exec(NULL) 直接返回"什么都没发生" */
        }

        case N_FN:
            /* 函数定义在执行前就登记好了，这里什么都不用做 */
            return f;

        case N_RETURN:
            /* return 只能出现在函数里 */
            if (call_depth == 0)
                fatal_at(n->tok, "return 只能在函数内部使用");
            f.returned = 1;
            f.value = n->a ? eval(n->a, e) : number(0);
            return f;   /* 这个 f 一路往上冒泡，直到函数调用处被接住 */

        case N_FOR: {
            /* 循环自带一个作用域：i 只在循环里活着 */
            Env local;
            env_init(&local, e);

            /* 第一段：初始化，只执行一次 */
            if (n->a) {
                f = exec(n->a, &local);
                if (f.returned) { env_free(&local); return f; }
            }

            /* 第二段：条件，每次迭代前检查一次 */
            while (!n->b || eval(n->b, &local).value != 0) {
                /* 安全阀：防止 while(true) 之类的死循环把解释器卡死 */
                if (++loop_steps > MAX_LOOP_STEPS) {
                    env_free(&local);
                    fatal_at(n->tok, "循环执行超过安全上限 %d 步", MAX_LOOP_STEPS);
                }
                f = exec(n->items[0], &local);          /* 循环体 */
                if (f.returned) { env_free(&local); return f; }
                if (n->c) {                               /* 第三段：更新 */
                    f = exec(n->c, &local);
                    if (f.returned) { env_free(&local); return f; }
                }
            }
            env_free(&local);
            return noflow();
        }

        default:
            fatal_at(n->tok, "这里不是一条语句");
    }
    return f;
}

/* ------------------------------------------------------------------ */
/* 读文件                                                              */
/* ------------------------------------------------------------------ */
static char *read_file(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) { perror(path); exit(1); }
    if (fseek(fp, 0, SEEK_END) != 0) { perror("fseek"); exit(1); }
    long len = ftell(fp);
    if (len < 0 || len > 4 * 1024 * 1024) { fprintf(stderr, "文件太大或无法读取\n"); exit(1); }
    rewind(fp);
    char *b = (char *)malloc((size_t)len + 1);
    if (!b) { perror("malloc"); exit(1); }
    size_t got = fread(b, 1, (size_t)len, fp);
    fclose(fp);
    b[got] = 0;
    return b;
}

/* 把字符串里的转义字符显示出来，方便看 token */
static void print_escaped(const char *s) {
    putchar('"');
    for (; *s; s++) {
        if      (*s == '\n') printf("\\n");
        else if (*s == '\t') printf("\\t");
        else if (*s == '\r') printf("\\r");
        else putchar(*s);
    }
    putchar('"');
}

/* ------------------------------------------------------------------ */
/* 三种模式：tokens / ast / run                                        */
/* 模式表在第一步就定型，后面每一步只往上加分支，不动 main 的骨架       */
/* ------------------------------------------------------------------ */
static int mode_tokens(const char *path) {
    src = read_file(path);
    lex();
    for (int i = 0; i < ntok; i++) {
        Token *t = &toks[i];
        printf("[%d:%d] %-8s ", t->line, t->col, type_name(t->type));
        if (t->type == T_STR) print_escaped(t->text);
        else printf("%s", t->text);
        if (t->type == T_NUM) printf("  (%g)", t->num);
        putchar('\n');
    }
    free((void *)src);
    return had_error ? 1 : 0;
}

static int mode_ast(const char *path) {
    src = read_file(path);
    lex();
    Node *prog = parse_program();
    printf("语法树：\n");
    dump_ast(prog, "");
    free((void *)src);
    return had_error ? 1 : 0;
}

static int mode_run(const char *path) {
    src = read_file(path);
    lex();
    Node *prog = parse_program();

    /* 先登记所有函数，再执行语句，这样函数之间可以互相调用 */
    register_functions(prog);

    Env global;
    env_init(&global, NULL);
    for (int i = 0; i < prog->count; i++) {
        Flow f = exec(prog->items[i], &global);
        if (f.returned)
            fatal_at(prog->items[i]->tok, "return 不能出现在函数外部");
    }
    env_free(&global);

    free((void *)src);
    return had_error ? 1 : 0;
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "用法: %s 脚本.ml [tokens|ast|run]\n", argv[0]);
        fprintf(stderr, "示例: fn square(x) { return x * x; } print(\"{square(5)}\\n\");\n");
        return 2;
    }
    const char *path = argv[1];
    const char *mode = (argc == 3) ? argv[2] : "run";

    if (!strcmp(mode, "tokens")) return mode_tokens(path);
    if (!strcmp(mode, "ast"))    return mode_ast(path);
    if (!strcmp(mode, "run"))    return mode_run(path);

    fprintf(stderr, "未知模式 '%s'，可用模式：tokens / ast / run\n", mode);
    return 2;
}
