#!/bin/sh
# 对照实验：如果第 5 步把 Env 的变量表写成固定数组，会发生什么？
#
#   ./run.sh
#
# 本目录下的 minilang.c 是第 11 步 minilang.c 的一份**改动过的副本**，
# 只改了 Env 的变量表这一处：
#
#   改动前（教程正文的写法）        改动后（本对照实验）
#   ─────────────────────────      ─────────────────────────
#   Var   *vars;                   Var    vars[2048];
#   int    count;                  int    count;
#   int    cap;                    int    cap;
#   struct Env *parent;            struct Env *parent;
#
# 其余代码一个字没动。
set -u
cd "$(dirname "$0")" || exit 1

CC=${CC:-cc}
CFLAGS=${CFLAGS:--std=c11 -Wall -Wextra -O2}

echo "== 编译对照版本（Env 用固定数组 Var vars[2048]） =="
$CC $CFLAGS minilang.c -lm -o .fixed_env || exit 1

echo "== 编译本步的正式版本（Env 用动态数组） =="
$CC $CFLAGS ../../minilang.c -lm -o .dynamic_env || exit 1

cat > .depth.ml <<'EOF'
fn depth(n) {
    if (n <= 0) { return 0; }
    return 1 + depth(n - 1);
}
print(depth(DEPTH));
EOF

for d in 5 8 10 20 100; do
    sed "s/DEPTH/$d/" .depth.ml > .depth_run.ml

    out=$(./.fixed_env .depth_run.ml 2>&1); st=$?
    if [ "$st" = 0 ]; then
        fixed="正常，输出 $out"
    elif [ "$st" = 139 ]; then
        fixed="💥 段错误（exit 139，栈溢出）"
    else
        fixed="退出码 $st：$out"
    fi

    out=$(./.dynamic_env .depth_run.ml 2>&1); st2=$?
    if [ "$st2" = 0 ]; then
        dyn="正常，输出 $out"
    else
        dyn="退出码 $st2：$out"
    fi

    printf "递归深度 %-4s  固定数组：%-28s  动态数组：%s\n" "$d" "$fixed" "$dyn"
done

rm -f .depth.ml .depth_run.ml .fixed_env .dynamic_env
