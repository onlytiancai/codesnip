#!/usr/bin/env bash
# build.sh - 检测 + 编译 + 打包成 .app + ad-hoc 签名 + 启动 GUI 一键脚本

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

ARCH=$(uname -m)
SWIFT_TARGET="${ARCH}-apple-macosx13.0"

echo "=== 阶段 1/4: 检测 Xcode Command Line Tools ==="
./detect.sh

echo
echo "=== 阶段 2/4: 编译 Swift + AppKit GUI，target = $SWIFT_TARGET ==="
mkdir -p build
if swiftc \
    -target "$SWIFT_TARGET" \
    -O \
    -o build/HelloWorld \
    -framework Cocoa \
    HelloWorld.swift; then
    echo "✓ 编译成功: build/HelloWorld"
else
    echo "✗ 编译失败"
    exit 1
fi

echo
echo "=== 阶段 3/4: 打包 .app bundle + ad-hoc 签名 ==="

rm -rf HelloWorld.app
mkdir -p HelloWorld.app/Contents/MacOS
mkdir -p HelloWorld.app/Contents/Resources

cp build/HelloWorld HelloWorld.app/Contents/MacOS/HelloWorld
cp Info.plist HelloWorld.app/Contents/Info.plist
cp Resources/AppIcon.icns HelloWorld.app/Contents/Resources/AppIcon.icns

# PkgInfo：8 字节老式 bundle 描述（APPL = application，???? 占位 creator）
printf 'APPL????' > HelloWorld.app/Contents/PkgInfo

# Ad-hoc 签名（--sign - 用设备身份，本机够用，无需 Apple Developer 账号）
if codesign --force --deep --sign - HelloWorld.app 2>&1; then
    echo "✓ Ad-hoc 签名成功"
else
    echo "✗ 签名失败"
    exit 1
fi

echo "✓ Bundle 结构:"
find HelloWorld.app -type f | sort | sed 's/^/    /'

echo
echo "=== 阶段 4/4: 启动 GUI ==="
echo "（关闭窗口或点「退出」按钮即可结束）"
open HelloWorld.app
sleep 2
GUI_PID=$(pgrep -f "HelloWorld.app/Contents/MacOS/HelloWorld" | head -n1 || true)
echo "GUI 进程 PID: ${GUI_PID:-未检测到}"

if [[ -n "$GUI_PID" ]]; then
    wait "$GUI_PID" 2>/dev/null || true
fi
echo "GUI 已退出。"
