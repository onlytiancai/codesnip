/*
 * MiniLang —— 词法分析器
 *
 * 职责：把源文件扫描成一串 Token，并提供错误报告功能。
 */
#include "minilang.h"

/* 全局状态：token 数组、扫描位置、错误计数 */
Token       toks[MAX_TOKENS];
int         ntok;
int         pos;
const char *src;
int         src_line, src_col;
int         had_error;
int         loop_steps;

/* ------------------------------------------------------------------ */
/* 错误处理                                                            */
/* ------------------------------------------------------------------ */

/* 打印出错的那一行源码，并在出错位置画一个指示符：
       3 | let x = price + 1;
         |         ^
   源码放在全局变量 src 里，所以这里不用再传一遍。 */
static void print_source_line(int line, int col) {
    if (!src || line <= 0) return;

    /* 找到第 line 行开头的位置 */
    const char *p = src;
    int cur = 1;
    while (cur < line && *p) {
        if (*p == '\n') cur++;
        p++;
    }
    if (cur != line) return;      /* 这一行不存在（比如行号是 0） */

    const char *end = strchr(p, '\n');
    size_t len = end ? (size_t)(end - p) : strlen(p);

    /* 太长的行只截取出错位置附近的一段，免得一行刷出几万个字符 */
    int start = 0;
    if (len > MAX_SHOW_COLS) {
        start = col - 1 - MAX_SHOW_COLS / 2;
        if (start < 0) start = 0;
        if (start > (int)len - MAX_SHOW_COLS) start = (int)len - MAX_SHOW_COLS;
    }
    int show = (int)len - start;
    if (show > MAX_SHOW_COLS) show = MAX_SHOW_COLS;

    fprintf(stderr, "%4d | ", line);
    if (start > 0) fputs("...", stderr);
    fprintf(stderr, "%.*s", show, p + start);
    if (start + show < (int)len) fputs("...", stderr);
    fputc('\n', stderr);

    /* 指示符要和上面的源码对齐：
       源码行的前缀是 "%4d | "，一共 7 列；指示符那行也必须是 7 列，
       否则 ^ 会比它指的那个字符偏左一格。 */
    int caret = col - 1 - start + (start > 0 ? 3 : 0);
    if (caret < 0) caret = 0;
    fprintf(stderr, "     | %*s^\n", caret, "");
}

void error_at(int line, int col, const char *fmt, ...) {
    va_list ap;
    fprintf(stderr, "错误 [%d:%d]: ", line, col);
    va_start(ap, fmt); vfprintf(stderr, fmt, ap); va_end(ap);
    fputc('\n', stderr);
    print_source_line(line, col);
    had_error = 1;
}

void fatal_at(Token t, const char *fmt, ...) {
    va_list ap;
    fprintf(stderr, "错误 [%d:%d]: ", t.line, t.col);
    va_start(ap, fmt); vfprintf(stderr, fmt, ap); va_end(ap);
    fputc('\n', stderr);
    print_source_line(t.line, t.col);
    had_error = 1;
    exit(1);
}

/* 复制一份字符串（参数名要保存到运行期，所以必须复制，不能指着源码） */
char *dupstr(const char *s) {
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
/* 扫描                                                                */
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
void lex(void) {
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
const char *type_name(TokType t) {
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
