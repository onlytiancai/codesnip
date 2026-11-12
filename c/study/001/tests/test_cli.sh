#!/bin/sh
# stats CLI 端到端测试（黑盒，调用编译好的 ./stats）
# 单独运行: ./tests/test_cli.sh    （需先执行 make）

set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
BIN="$ROOT/stats"
DATA="$ROOT/tests/data"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT HUP INT TERM

[ -x "$BIN" ] || { echo "找不到 $BIN，请先运行 make" >&2; exit 1; }

pass=0
fail=0

ok()  { pass=$((pass + 1)); printf '  ok   %s\n' "$1"; }
bad() { fail=$((fail + 1)); printf '  FAIL %s\n' "$1" >&2; }

assert_eq() {  # 名称 期望 实际
    if [ "$2" = "$3" ]; then
        ok "$1"
    else
        bad "$1（期望 [$2]，实际 [$3]）"
    fi
}

assert_has() {  # 名称 关键字 文件
    if grep -q -- "$2" "$3"; then
        ok "$1"
    else
        bad "$1（输出里没有 '$2'）"
    fi
}

expect_fail() {  # 名称 期望的错误信息 命令 [参数...]
    name=$1 needle=$2
    shift 2
    if "$@" >"$TMP/out" 2>"$TMP/err"; then
        bad "$name（本应失败却成功了）"
    elif grep -q -- "$needle" "$TMP/err"; then
        ok "$name"
    else
        bad "$name（错误信息里没有 '$needle'）"
    fi
}

# 用 awk 的 %d 精确生成 n 行整数（macOS 的 seq 在 >=1000000 时会多输出一行
# 并写成 1e+06，格式不可控）。
gen() {
    awk -v n="$1" 'BEGIN { for (i = 1; i <= n; i++) printf "%d\n", i }'
}

expect_pipe_fail() {  # 名称 期望的错误信息 stdin 内容
    name=$1 needle=$2
    if printf '%s' "$3" | "$BIN" >"$TMP/out" 2>"$TMP/err"; then
        bad "$name（本应失败却成功了）"
    elif grep -q -- "$needle" "$TMP/err"; then
        ok "$name"
    else
        bad "$name（错误信息里没有 '$needle'）"
    fi
}

echo "stats CLI 端到端测试"

# --- 常规统计 ---
got=$(seq 1 100 | "$BIN" -c | tail -n 1)
assert_eq "1..100 默认 linear 百分位" \
    "100,1,100,50.5,50.5,85.15,90.1,95.05,99.01" "$got"

got=$(seq 1 100 | "$BIN" -m nearest -c | tail -n 1)
assert_eq "1..100 nearest 百分位" \
    "100,1,100,50.5,50.5,85,90,95,99" "$got"

if "$BIN" -f "$DATA/sample.txt" >"$TMP/sample.out" 2>"$TMP/err"; then
    if diff -u "$DATA/expected.txt" "$TMP/sample.out" >"$TMP/diff" 2>&1; then
        ok "样例文件输出与 expected.txt 一致"
    else
        bad "样例文件输出与 expected.txt 一致"
        cat "$TMP/diff" >&2
    fi
else
    bad "样例文件输出与 expected.txt 一致（程序退出失败）"
fi

printf '1\n2\n3\n' | "$BIN" -p 50,99.9 -c >"$TMP/hdr"
got=$(head -n 1 "$TMP/hdr")
assert_eq "自定义百分位的 CSV 表头" \
    "count,min,max,mean,median,p50,p99.9" "$got"

got=$(printf '42\n' | "$BIN" -m nearest -p 0,50,100 -c | tail -n 1)
assert_eq "单元素 + p0/p50/p100" "1,42,42,42,42,42,42,42" "$got"

# --- 输入格式 ---
literal='100\n50\n22\n46\n73\n'
got=$(printf '%s' "$literal" | "$BIN" -c | tail -n 1)
assert_eq "字面 \\n 转义（bash echo 场景）" "5,22,100,58.2,50,83.8,89.2,94.6,98.92" "$got"

got=$(printf '1, 2; 3\t4 # 注释\n5\n' | "$BIN" -c | tail -n 1)
assert_eq "逗号/分号/制表符/注释混用" "5,1,5,3,3,4.4,4.6,4.8,4.96" "$got"

# --- 错误处理 ---
expect_pipe_fail "非法数值报错" "无法解析" '1 2 oops'
expect_pipe_fail "nan 报错" "有限" '1 nan 2'
expect_pipe_fail "空输入报错" "没有有效数值" ''
expect_fail "越界百分位报错" "非法" "$BIN" -p 101 1 2
expect_fail "非法 -m 报错" "只接受" "$BIN" -m bogus 1 2
expect_fail "非法 -n 报错" "非负整数" "$BIN" -n abc 1 2
expect_fail "负的 -n 报错" "非负整数" "$BIN" -n -1 1 2

# --- 数量上限 ---
expect_fail "超过 -n 3 上限报错" "超过上限 3" "$BIN" -n 3 1 2 3 4
expect_fail "-n 出现在数值之后也生效" "超过上限 3" "$BIN" 1 2 3 4 -n 3
got=$("$BIN" -n 3 1 2 3 -c | tail -n 1 | cut -d, -f1)
assert_eq "恰好达到 -n 3 上限" "3" "$got"

"$BIN" -h >"$TMP/help"
assert_has "帮助里说明 --max 选项" "--max" "$TMP/help"
assert_has "帮助里说明默认上限 1000000" "默认 1000000" "$TMP/help"

# 上限边界用 -n 1000 缩小规模验证，不必真的生成 100 万数据；
# 默认值 1000000 由 test_stats.c 的常量断言 + 上面的帮助文本共同保证。
echo "  正在验证上限边界（-n 1000 缩小规模）..."
if gen 1000 | "$BIN" -n 1000 -c >"$TMP/big" 2>"$TMP/err"; then
    got=$(head -n 2 "$TMP/big" | tail -n 1 | cut -d, -f1)
    assert_eq "恰好 1000 个数可以通过" "1000" "$got"
else
    bad "恰好 1000 个数可以通过（程序退出失败）"
fi

if gen 1001 | "$BIN" -n 1000 -c >"$TMP/big2" 2>"$TMP/err2"; then
    bad "第 1001 个数应触发上限报错"
else
    assert_has "第 1001 个数触发上限报错" "超过上限 1000" "$TMP/err2"
fi

if gen 1001 | "$BIN" -n 0 -c >"$TMP/big3" 2>"$TMP/err3"; then
    got=$(head -n 2 "$TMP/big3" | tail -n 1 | cut -d, -f1)
    assert_eq "-n 0 可解除上限" "1001" "$got"
else
    bad "-n 0 可解除上限（程序退出失败）"
fi

# 不加 -n 时默认值应明显宽松于 1000
if gen 1001 | "$BIN" -c >"$TMP/default" 2>"$TMP/err4"; then
    got=$(head -n 2 "$TMP/default" | tail -n 1 | cut -d, -f1)
    assert_eq "默认上限下 1001 个数正常通过" "1001" "$got"
else
    bad "默认上限下 1001 个数正常通过（程序退出失败）"
fi

printf 'CLI 测试: %d 通过，%d 失败\n' "$pass" "$fail"
if [ "$fail" -ne 0 ]; then exit 1; fi
exit 0
