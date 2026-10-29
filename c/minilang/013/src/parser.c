/*
 * MiniLang —— 语法分析器
 *
 * 职责：把 token 序列变成抽象语法树（递归下降）。
 */
#include "minilang.h"

static Node *statement(void);

/* 当前读到的 token */
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

/* ------------------------------------------------------------------ */
/* 表达式                                                              */
/* ------------------------------------------------------------------ */

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
Node *expression(void) { return logical_or(); }

/* ------------------------------------------------------------------ */
/* 语句                                                                */
/* ------------------------------------------------------------------ */

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

    /* 其它都是表达式语句：表达式; */
    Node *n = node(N_EXPR, t);
    n->a = expression();
    expect(T_SEMI, "语句末尾需要 ';'");
    return n;
}

/* 整个程序：一串语句 */
Node *parse_program(void) {
    Node *n = node(N_BLOCK, toks[0]);
    while (cur()->type != T_EOF) {
        if (n->count >= 128) fatal_at(*cur(), "顶层语句过多（上限 128）");
        n->items[n->count++] = statement();
    }
    return n;
}
