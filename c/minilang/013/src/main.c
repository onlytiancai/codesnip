/*
 * MiniLang —— 入口
 *
 * 职责：读文件、模式分发、main。
 */
#include "minilang.h"

/* ------------------------------------------------------------------ */
/* 读文件                                                              */
/* ------------------------------------------------------------------ */
char *read_file(const char *path) {
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
/* 模式表从第一步就定型，后面每一步只往上加分支，不动 main 的骨架       */
/* ------------------------------------------------------------------ */
int mode_tokens(const char *path) {
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

int mode_ast(const char *path) {
    src = read_file(path);
    lex();
    Node *prog = parse_program();
    printf("语法树：\n");
    dump_ast(prog, "");
    free((void *)src);
    return had_error ? 1 : 0;
}

int mode_run(const char *path) {
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
