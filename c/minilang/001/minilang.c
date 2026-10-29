/*
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
#include <stdarg.h>

#define MAX_TOKENS 20000
#define MAX_TEXT   4096

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
static const char *src;          /* 整份源代码 */
static int   src_line, src_col;  /* 扫描器当前所在的位置 */
static int   had_error;          /* 出过错没有 */

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

/* token 类型的中文/英文名字，打印时用 */
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
    (void)path;
    fprintf(stderr, "语法分析尚未实现，请先用 tokens 模式查看词法结果\n");
    return 1;
}

static int mode_run(const char *path) {
    (void)path;
    fprintf(stderr, "解释执行尚未实现，请先用 tokens 模式查看词法结果\n");
    return 1;
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
