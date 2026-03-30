# EN-READER3

AI-powered English article video generator with Chinese narration.

## 命令行参数介绍

```bash
# Phase 1: 分段 - 将文章分成段落
pnpm exec tsx src/index.ts --input input/test-short.txt --phase 1

# Phase 2: 生成 AI 脚本（需要 phase 1 输出）
pnpm exec tsx src/index.ts --input input/test-short.txt --phase 2

# Phase 3: 生成音频和 HTML（需要 phase 2 输出）
pnpm exec tsx src/index.ts --input input/test-short.txt --phase 3

# Phase 4: 生成 ASS 字幕文件（需要 phase 3 输出）
pnpm exec tsx src/index.ts --input input/test-short.txt --phase 4

# Phase 5: 截图 + FFmpeg 合成 MP4（需要 phase 3、4 输出，自动检查 ASS）
pnpm exec tsx src/index.ts --input input/test-short.txt --phase 5

# Phase 6: 连接所有 MP4 为最终视频（需要 phase 5 输出）
pnpm exec tsx src/index.ts --input input/test-short.txt --phase 6 --output ./output/final.mp4

# 完整管道（默认 - 运行所有阶段）
pnpm exec tsx src/index.ts --input input/test-short.txt --output ./output/video.mp4
```

## 各阶段输出文件

### Phase 1 - 分段
| 输出文件 | 说明 |
|---------|------|
| `output/segments.json` | 文章分段结果，包含各段原文 |

### Phase 2 - AI 脚本生成
| 输出文件 | 说明 |
|---------|------|
| `output/article-script.json` | AI 生成的完整脚本，包含 intro、segments、outro |

### Phase 3 - 音频和 HTML
每个 section 生成以下文件：

| 路径 | 说明 |
|------|------|
| `output/segments/{section}/section-{id}.mp3` | TTS 音频文件 |
| `output/segments/{section}/section-{id}-words.txt` | 词级时间戳 |
| `output/segments/{section}/slide-{id}-{part}.html` | HTML 幻灯片 |

其中 `{section}` 包括：
- `intro/` - 导言部分
- `segment-{n}/part-{m}/` - 各段落各部分
- `outro/` - 结尾部分

### Phase 4 - ASS 字幕文件
每个 section 生成以下文件：

| 路径 | 说明 |
|------|------|
| `output/segments/{section}/section-{id}.ass` | Karaoke 字幕文件 |

Phase 4 还会生成 `output/phase-4-result.json` 记录各字幕路径。

### Phase 5 - 截图和 MP4 视频
每个 section 生成以下文件：

| 路径 | 说明 |
|------|------|
| `output/segments/{section}/slide-{id}.png` | 幻灯片截图（intro/outro） |
| `output/segments/{section}/slide-{id}-{part}.png` | 幻灯片截图（segment） |
| `output/segments/{section}/{id}.mp4` | 合成后的 MP4 视频（带字幕） |

Phase 5 会在开始时自动运行 `--check-ass` 检查所有 ASS 文件，如有错误则中断执行。

Phase 5 还会生成 `output/phase-5-result.json` 记录各视频路径。

### Phase 6 - 最终视频拼接
读取 `output/phase-5-result.json` 中的所有 MP4 片段，拼接为最终视频输出。

## 其他命令

```bash
# 评估 AI 生成的脚本质量
pnpm exec tsx src/index.ts --evaluate

# 单独检查 ASS 字幕文件
pnpm exec tsx src/index.ts --check-ass
```