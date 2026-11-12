#!/usr/bin/env bash
# detect.sh - 检测本机 Xcode Command Line Tools 是否可用
# 用法：./detect.sh      单独跑（退出码 0 = 全部可用）

set -euo pipefail

# 颜色（仅当 stdout 是 TTY 时启用）
if [[ -t 1 ]]; then
    C_RED='\033[0;31m'
    C_GREEN='\033[0;32m'
    C_CYAN='\033[0;36m'
    C_YELLOW='\033[0;33m'
    C_BOLD='\033[1m'
    C_RST='\033[0m'
else
    C_RED=''; C_GREEN=''; C_CYAN=''; C_YELLOW=''; C_BOLD=''; C_RST=''
fi

pass() { printf "${C_GREEN}\xe2\x9c\x93${C_RST} %s\n" "$1"; }
fail() { printf "${C_RED}\xe2\x9c\x97${C_RST} %s\n" "$1"; }
info() { printf "${C_CYAN}\xe2\x84\xb9${C_RST} %s\n" "$1"; }
warn() { printf "${C_YELLOW}!${C_RST} %s\n" "$1"; }

printf "${C_BOLD}=== Xcode Command Line Tools 检测 ===${C_RST}\n"

FAILED=0

# 1. xcode-select -p
if CLT_PATH=$(xcode-select -p 2>/dev/null); then
    pass "xcode-select 路径:    $CLT_PATH"
else
    fail "xcode-select 无法定位 CLT"
    warn "安装命令: xcode-select --install"
    FAILED=1
fi

# 2. swiftc 在 PATH
if SWIFTC_PATH=$(command -v swiftc); then
    pass "swiftc:                $SWIFTC_PATH"
else
    fail "swiftc 不在 PATH"
    FAILED=1
fi

# 3. xcrun --find swiftc
if XCRUN_SWIFTC=$(xcrun --find swiftc 2>/dev/null); then
    pass "xcrun swiftc:          $XCRUN_SWIFTC"
else
    fail "xcrun 找不到 swiftc（CLT 可能损坏）"
    FAILED=1
fi

# 4. macOS SDK 路径
if SDK_PATH=$(xcrun --show-sdk-path --sdk macosx 2>/dev/null); then
    pass "macOS SDK 路径:        $SDK_PATH"
else
    fail "macOS SDK 路径不可用"
    FAILED=1
fi

# 5. macOS SDK 版本（仅打印）
if SDK_VER=$(xcrun --show-sdk-version --sdk macosx 2>/dev/null); then
    info "macOS SDK 版本:        $SDK_VER"
else
    warn "无法读取 macOS SDK 版本"
fi

# 6. Swift 版本（仅打印）
if SWIFT_VER=$(swift --version 2>&1 | head -n1); then
    info "Swift:                 $SWIFT_VER"
else
    warn "无法读取 Swift 版本"
fi

echo
if [[ $FAILED -eq 0 ]]; then
    printf "${C_GREEN}${C_BOLD}=== 全部组件可用 ===${C_RST}\n"
    exit 0
else
    printf "${C_RED}${C_BOLD}=== 检测失败 ===${C_RST}\n"
    exit 1
fi
