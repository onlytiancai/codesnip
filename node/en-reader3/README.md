# EN-READER3

AI-powered English article video generator with Chinese narration.

## 命令行参数介绍

```bash
# Phase 1: 分段 - 将文章分成段落
pnpm dev -- --input input/test-short.txt --phase 1

# Phase 2: 生成 AI 脚本（需要 phase 1 输出）
pnpm dev -- --input input/test-short.txt --phase 2

# Phase 3: 生成 TTS 音频（需要 phase 2 输出）
pnpm dev -- --input input/test-short.txt --phase 3

# Phase 4: 生成 ASS 字幕文件（需要 phase 3 输出）
pnpm dev -- --input input/test-short.txt --phase 4

# Phase 5: 生成 HTML 幻灯片、截图和 MP4 视频（需要 phase 3、4 输出）
pnpm dev -- --input input/test-short.txt --phase 5

# Phase 6: 连接所有 MP4 为最终视频（需要 phase 5 输出）
pnpm dev -- --input input/test-short.txt --phase 6 --output ./output/final.mp4

# 完整管道（默认 - 运行所有阶段）
pnpm dev -- --input input/test-short.txt --output ./output/video.mp4
```

## 其他命令

```bash
# 评估 AI 生成的脚本质量
pnpm dev -- --input input/test-short.txt --evaluate

# 单独检查 ASS 字幕文件
pnpm dev -- --check-ass
```