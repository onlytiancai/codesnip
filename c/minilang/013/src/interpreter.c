/*
 * MiniLang —— 解释执行
 *
 * 职责：作用域与变量、表达式求值、语句执行、内置函数、函数调用。
 */
#include "minilang.h"

/* ------------------------------------------------------------------ */
/* 作用域与变量                                                        */
/* ------------------------------------------------------------------ */
void env_init(Env *e, Env *parent) {
    e->vars   = NULL;
    e->count  = 0;
    e->cap    = 0;
    e->parent = parent;
}

void env_free(Env *e) {
    free(e->vars);
    e->vars  = NULL;
    e->count = 0;
    e->cap   = 0;
}

/* 在当前作用域里加一个变量。变量表是动态数组：
   用多少占多少，不用一开始就开 2048 个槽位。 */
static void env_push(Env *e, const char *name, double v) {
    if (e->count >= MAX_VARS)
        fatal_at((Token){.line = 0, .col = 0}, "变量数量超出上限");
    if (e->count == e->cap) {
        int ncap = e->cap ? e->cap * 2 : 8;
        Var *nv = (Var *)realloc(e->vars, (size_t)ncap * sizeof(Var));
        if (!nv) { perror("realloc"); exit(1); }
        e->vars = nv;
        e->cap  = ncap;
    }
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

/* ------------------------------------------------------------------ */
/* 内置函数表                                                          */
/*                                                                     */
/* 表里放的是"数字进、数字出"的纯函数。print 不在表里——它要输出，   */
/* 而且还要访问作用域去查模板里的变量，需求不一样。                   */
/* ------------------------------------------------------------------ */
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
void register_functions(Node *root) {
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
/* 求值与执行                                                          */
/* ------------------------------------------------------------------ */

/* 造一个数字值 */
static Value number(double x) {
    Value v;
    v.value = x;
    v.is_string = 0;
    v.string[0] = '\0';
    return v;
}

/* 什么都不返回 */
static Flow noflow(void) {
    Flow f;
    f.returned = 0;
    f.value = number(0);
    return f;
}

/* exec 定义在 eval 后面，但 eval 里要调用它：
   C 语言要求"先声明后使用"，头文件里已经有原型了。 */

/* 求值：把一棵表达式树算成一个值。
   递归的结构和语法分析器正好相反：分析时"从左到右展开"，
   求值时"从叶子往回算"。 */
Value eval(Node *n, Env *e) {
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

            return flow.returned ? flow.value : number(0);
        }

        default:
            /* 语句节点不能出现在表达式里 */
            fatal_at(n->tok, "此处不是表达式");
    }
    return number(0);
}

/* 执行一条语句。返回值是一个 Flow，告诉调用者"要不要带着结果往上传"。 */
Flow exec(Node *n, Env *e) {
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

        default:
            fatal_at(n->tok, "这里不是一条语句");
    }
    return f;
}
