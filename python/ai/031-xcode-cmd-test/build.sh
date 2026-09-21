#!/usr/bin/env bash
# build.sh - 检测 + 编译 + 启动 GUI 一键脚本

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 按架构选 target。Apple 的 macOS SDK 是 universal 的，
# ${ARCH}-apple-macosx13.0 形式对 arm64 和 x86_64 都成立，无需 case 分支
ARCH=$(uname -m)
SWIFT_TARGET="${ARCH}-apple-macosx13.0"

echo "=== 阶段 1/3: 检测 Xcode Command Line Tools ==="
./detect.sh

echo
echo "=== 阶段 2/3: 编译 Swift + AppKit GUI，target = $SWIFT_TARGET ==="
if swiftc \
    -target "$SWIFT_TARGET" \
    -O \
    -o HelloWorld \
    -framework Cocoa \
    HelloWorld.swift; then
    echo "✓ 编译成功: ./HelloWorld"
else
    echo "✗ 编译失败"
    exit 1
fi

echo
echo "=== 阶段 3/3: 启动 GUI ==="
echo "（关闭窗口或点「退出」按钮即可结束）"
./HelloWorld &
GUI_PID=$!
echo "GUI 进程 PID: $GUI_PID"
wait "$GUI_PID" || true
echo "GUI 已退出。"
