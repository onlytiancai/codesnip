#!/bin/sh
# MiniLang 教程 —— 第 13 步的测试脚本
#
# 这一步有两种实现，必须证明它们的行为完全一致：
#   1) minilang.c      单文件版（和第 12 步一模一样）
#   2) src/*.c + Makefile   拆成多文件之后的版本
#
# 所以本脚本会：编译两个版本 → 用同一套用例分别跑 → 再逐字节对拍。

set -u
cd "$(dirname "$0")" || exit 1

CC=${CC:-cc}
CFLAGS=${CFLAGS:--std=c11 -Wall -Wextra -O2}
BIN_SINGLE=.minilang.single
BIN_MULTI=.minilang.multi

DEFAULT_MODE=run
[ -f tests/mode ] && DEFAULT_MODE=$(cat tests/mode)

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT INT TERM

pass=0
fail=0

# ---- 编译两个版本 ----------------------------------------------------
echo "▶ 编译单文件版"
if ! $CC $CFLAGS minilang.c -lm -o "$BIN_SINGLE"; then
    echo "❌ 编译失败"
    exit 1
fi
echo "▶ 编译多文件版（make）"
if ! make -s clean >/dev/null 2>&1 || ! make -s; then
    echo "❌ 编译失败"
    exit 1
fi
cp minilang "$BIN_MULTI"

# ---- 跑用例 ----------------------------------------------------------
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
    diff -u "$exp_out" "$TMP/out" >"$TMP/d1" || ok=0
    diff -u "$exp_err" "$TMP/err" >"$TMP/d2" || ok=0
    [ "$st" = "$exp_st" ] || ok=0

    if [ "$ok" = 1 ]; then
        pass=$((pass + 1))
        printf "  ✅ %-28s [%s]\n" "$(basename "$file")" "$mode"
    else
        fail=$((fail + 1))
        printf "  ❌ %-28s [%s]\n" "$(basename "$file")" "$mode"
        [ -s "$TMP/d1" ] && { echo "     --- 标准输出不符 ---"; sed 's/^/     /' "$TMP/d1"; }
        [ -s "$TMP/d2" ] && { echo "     --- 标准错误不符 ---"; sed 's/^/     /' "$TMP/d2"; }
        [ "$st" != "$exp_st" ] && echo "     --- 退出码：期望 $exp_st，实际 $st ---"
    fi
}

echo "▶ 用例（单文件版，模式：${DEFAULT_MODE}）"
for f in tests/*.ml; do
    [ -e "$f" ] || continue
    run_case "./$BIN_SINGLE" "$f"
done

echo "▶ 用例（多文件版，模式：${DEFAULT_MODE}）"
for f in tests/*.ml; do
    [ -e "$f" ] || continue
    run_case "./$BIN_MULTI" "$f"
done

# ---- 两个版本对拍 ----------------------------------------------------
echo "▶ 对拍：单文件版 vs 多文件版"
mismatch=0
for f in tests/*.ml demo.ml; do
    [ -e "$f" ] || continue
    mode=$DEFAULT_MODE
    base=${f%.ml}
    [ -f "$base.mode" ] && mode=$(cat "$base.mode")

    s1=0; "./$BIN_SINGLE" "$f" "$mode" >"$TMP/a.out" 2>"$TMP/a.err" || s1=$?
    s2=0; "./$BIN_MULTI"  "$f" "$mode" >"$TMP/b.out" 2>"$TMP/b.err" || s2=$?

    if ! cmp -s "$TMP/a.out" "$TMP/b.out" || ! cmp -s "$TMP/a.err" "$TMP/b.err" || [ "$s1" != "$s2" ]; then
        echo "  ❌ $(basename "$f")：两个版本输出不一致"
        mismatch=$((mismatch + 1))
    fi
done
if [ "$mismatch" = 0 ]; then
    echo "  ✅ 两个版本输出逐字节一致"
    pass=$((pass + 1))
else
    fail=$((fail + 1))
fi

# ---- demo 冒烟测试 --------------------------------------------------
echo "▶ demo.ml"
if "./$BIN_MULTI" demo.ml "$DEFAULT_MODE" >"$TMP/out" 2>"$TMP/err"; then
    sed 's/^/  | /' "$TMP/out"
    pass=$((pass + 1))
else
    fail=$((fail + 1))
    echo "  ❌ demo.ml 运行失败："
    sed 's/^/     /' "$TMP/err"
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
