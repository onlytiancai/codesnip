#!/bin/sh
# MiniLang 教程 —— 本步的"一键编译 + 跑全部测试"
#
#   ./run.sh
#
# 测试用例放在 tests/ 目录，命名规则：
#   xxx.ml        测试程序（MiniLang 源码）
#   xxx.stdout    期望的标准输出（不写就当作空）
#   xxx.stderr    期望的标准错误（不写就当作空）
#   xxx.status    期望的退出码（不写就当作 0）
#   xxx.mode      本用例用哪个模式跑（不写就用 tests/mode 里的默认模式）
#
# 退出码：全部通过是 0，有任何失败是 1。

set -u
cd "$(dirname "$0")" || exit 1

CC=${CC:-cc}
CFLAGS=${CFLAGS:--std=c11 -Wall -Wextra -O2}
BIN=minilang

# ---- 编译 ------------------------------------------------------------
if [ -f Makefile ]; then
    make -s clean >/dev/null 2>&1
    if ! make -s; then
        echo "❌ 编译失败"
        exit 1
    fi
    cp minilang "$BIN"
else
    if ! $CC $CFLAGS minilang.c -lm -o "$BIN"; then
        echo "❌ 编译失败"
        exit 1
    fi
fi

# ---- 准备 -----------------------------------------------------------
DEFAULT_MODE=run
[ -f tests/mode ] && DEFAULT_MODE=$(cat tests/mode)

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT INT TERM

pass=0
fail=0

run_case() {
    bin=$1
    file=$2
    base=${file%.ml}

    mode=$DEFAULT_MODE
    [ -f "$base.mode" ] && mode=$(cat "$base.mode")

    st=0
    "$bin" "$file" "$mode" >"$TMP/out" 2>"$TMP/err" || st=$?

    exp_out=/dev/null; [ -f "$base.stdout" ] && exp_out=$base.stdout
    exp_err=/dev/null; [ -f "$base.stderr" ] && exp_err=$base.stderr
    exp_st=0;        [ -f "$base.status" ] && exp_st=$(cat "$base.status")

    ok=1
    if ! diff -u "$exp_out" "$TMP/out" >"$TMP/d1"; then ok=0; fi
    if ! diff -u "$exp_err" "$TMP/err" >"$TMP/d2"; then ok=0; fi
    if [ "$st" != "$exp_st" ]; then ok=0; fi

    if [ "$ok" = 1 ]; then
        pass=$((pass + 1))
        printf "  ✅ %-28s [%s]\n" "$(basename "$file")" "$mode"
    else
        fail=$((fail + 1))
        printf "  ❌ %-28s [%s]\n" "$(basename "$file")" "$mode"
        [ -s "$TMP/d1" ] && { echo "     --- 标准输出不符 ---"; sed 's/^/     /' "$TMP/d1"; }
        [ -s "$TMP/d2" ] && { echo "     --- 标准错误不符 ---"; sed 's/^/     /' "$TMP/d2"; }
        [ "$st" != "$exp_st" ] && echo "     --- 退出码：期望 ${exp_st}，实际 $st ---"
    fi
}

echo "▶ 用例（模式：${DEFAULT_MODE}）"
for f in tests/*.ml; do
    [ -e "$f" ] || continue
    run_case "./$BIN" "$f"
done

# ---- demo 冒烟测试 --------------------------------------------------
echo "▶ demo.ml"
if [ -f demo.ml ]; then
    if "./$BIN" demo.ml "$DEFAULT_MODE" >"$TMP/out" 2>"$TMP/err"; then
        sed 's/^/  | /' "$TMP/out"
        pass=$((pass + 1))
    else
        fail=$((fail + 1))
        echo "  ❌ demo.ml 运行失败："
        sed 's/^/     /' "$TMP/err"
    fi
else
    echo "  （本步没有 demo.ml）"
fi

# ---- 汇总 -----------------------------------------------------------
echo
if [ "$fail" = 0 ]; then
    echo "✅ 全部通过（$pass 项）"
    exit 0
else
    echo "❌ 有 $fail 项失败，共 $((pass + fail)) 项"
    exit 1
fi
