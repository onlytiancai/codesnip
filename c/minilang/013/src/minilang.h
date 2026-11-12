/*
 * MiniLang —— 用 C 实现一门小型解释型语言
 *
 * 这是拆成多文件之后的公共头文件：
 * 所有模块共用的常量、类型和函数声明都放在这里。
 *
 * 原来单文件里的 static 变量和函数，在这里变成"头文件里 extern 声明、
 * 某一个 .c 文件里定义"；只在本文件里用的辅助函数仍然保持 static。
 */
#ifndef MINILANG_H
#define MINILANG_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <stdarg.h>

/* ------------------------------------------------------------------ */
/* 全局上限                                                            */
/* ------------------------------------------------------------------ */
#define MAX_TOKENS     20000
#define MAX_NODES      20000
#define MAX_TEXT       4096
#define MAX_PARAMS     32
#define MAX_VARS       2048
#define MAX_NAME       128
#define MAX_FUNCS      256
#define MAX_CALL_DEPTH 128
#define MAX_LOOP_STEPS 1000000
#define MAX_SHOW_COLS 160     /* 出错时一行最多显示多少列源码 */

/* ------------------------------------------------------------------ */
/* Token                                                               */
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
    TokType type;
    char    text[MAX_TEXT];
    double  num;
    int     line, col;
} Token;

/* ------------------------------------------------------------------ */
/* AST                                                                 */
/* ------------------------------------------------------------------ */
typedef enum {
    N_NUM, N_STR, N_VAR, N_UNARY, N_BINARY, N_CALL,
    N_BLOCK, N_LET, N_ASSIGN, N_EXPR, N_IF, N_FOR, N_FN, N_RETURN
} NodeType;

typedef struct Node Node;
struct Node {
    NodeType type;
    Token   tok;
    double  num;
    char    text[MAX_TEXT];
    int     op;
    Node   *a, *b, *c;
    Node   *items[128];
    int     count;
    char   *params[MAX_PARAMS];
    int     nparams;
};

/* ------------------------------------------------------------------ */
/* 运行时的类型                                                        */
/* ------------------------------------------------------------------ */
typedef struct {
    double value;
    int    is_string;
    char   string[MAX_TEXT];
} Value;

typedef struct {
    int   returned;
    Value value;
} Flow;

typedef struct {
    char   name[MAX_NAME];
    double value;
} Var;

typedef struct Env {
    Var   *vars;
    int    count;
    int    cap;
    struct Env *parent;
} Env;

typedef double (*BuiltinFn)(double *args, Token tok);

typedef struct {
    const char *name;
    int         min_args;
    int         max_args;
    BuiltinFn   fn;
} Builtin;

typedef struct {
    char  name[MAX_NAME];
    Node *decl;
} Func;

/* ------------------------------------------------------------------ */
/* lexer.c                                                             */
/* ------------------------------------------------------------------ */
extern Token toks[MAX_TOKENS];
extern int   ntok;
extern int   pos;
extern const char *src;
extern int   src_line, src_col;
extern int   had_error;
extern int   loop_steps;

void        lex(void);
void        error_at(int line, int col, const char *fmt, ...);
void        fatal_at(Token t, const char *fmt, ...);
const char *type_name(TokType t);
char       *dupstr(const char *s);

/* ------------------------------------------------------------------ */
/* ast.c                                                               */
/* ------------------------------------------------------------------ */
Node *node(NodeType ty, Token t);
void  dump_ast(Node *n, const char *prefix);

/* ------------------------------------------------------------------ */
/* parser.c                                                            */
/* ------------------------------------------------------------------ */
Node *expression(void);
Node *parse_program(void);

/* ------------------------------------------------------------------ */
/* interpreter.c                                                       */
/* ------------------------------------------------------------------ */
Value eval(Node *n, Env *e);
Flow  exec(Node *n, Env *e);
void  register_functions(Node *root);
void  env_init(Env *e, Env *parent);
void  env_free(Env *e);

/* ------------------------------------------------------------------ */
/* main.c                                                              */
/* ------------------------------------------------------------------ */
char *read_file(const char *path);
int   mode_tokens(const char *path);
int   mode_ast(const char *path);
int   mode_run(const char *path);

#endif /* MINILANG_H */
