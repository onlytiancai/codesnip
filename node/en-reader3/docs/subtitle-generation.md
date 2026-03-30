# 字幕生成原理

## 概述

本文档介绍如何将旁白脚本（Narration Script）转换为 ASS 格式字幕，并烧录到 MP4 视频中。

## 整体流程

```
文本脚本 → Edge TTS (生成音频 + 单词时间戳) → 文本分段 → 时间映射 → ASS字幕 → FFmpeg烧录
```

## 1. Edge TTS 单词时间戳

使用 Edge TTS 的 `WordBoundary` 事件获取每个单词的精确时间：

```python
async def get_word_timings(text):
    words = []
    communicate = edge_tts.Communicate(text, boundary="WordBoundary")
    async for chunk in communicate.stream():
        if chunk["type"] == "WordBoundary":
            offset_seconds = chunk["offset"] / 10000000  # 100纳秒转秒
            duration_seconds = chunk["duration"] / 10000000
            words.append({
                "text": chunk["text"],
                "start": offset_seconds,
                "end": offset_seconds + duration_seconds
            })
    return words
```

输出格式：`start:end:text`（每行一个单词）

## 2. 文本分段

根据标点符号将长文本分成短字幕段：

### 中文规则
- 句子结束符：`。！？`
- 逗号分隔：`，`、`、`
- 每段最多 30 字符
- 如果逗号分隔后少于 5 字，则合并到下一段

### 英文规则
- 句子结束符：`.!?`
- 逗号分隔：`,`
- 每段最多 15 单词

### 示例

```
原文：同学们，今天我们要读一篇英语故事，主人公大家肯定都不陌生。
分段：
1. 同学们，今天我们要读一篇英语故事，
2. 主人公大家肯定都不陌生。
```

## 3. ASS 字幕格式

### 文件结构

```ass
[Script Info]
ScriptType: v4.00+
PlayResX: 1080          # 视频分辨率宽
PlayResY: 1920          # 视频分辨率高

[V4+ Styles]            # 样式定义
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Heiti SC,45,&Hffffff,&Hffffff,&H0,&H0,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1

[Events]                 # 字幕事件
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,00:00:00.15,00:00:02.35,Default,,0,0,0,,这是第一行\N这是第二行
```

### 关键参数

| 参数 | 含义 |
|------|------|
| PlayResX/PlayResY | 视频分辨率，字幕坐标基于此 |
| Fontname | 字体名，"Heiti SC"=黑体 |
| Fontsize | 字体大小（像素） |
| PrimaryColour | 文字颜色，&Hffffff=白色 |
| Alignment | 位置：2=底部居中，9=顶部居中 |
| MarginV | 距底部边距 |
| \N | 换行符（ASS专用） |

### Alignment 位置对照

```
 7   8   9    ← 顶部
 4   5   6    ← 中部
 1   2   3    ← 底部（最常用）
```

## 4. 时间映射

### 问题

Edge TTS 提供的是**单词级别**的时间戳，但我们要显示的是**分段后**的字幕。需要将分段文本映射到正确的时间范围。

### 映射算法

1. 将 TTS 单词列表按时间顺序展开
2. 对每个字幕分段：
   - 找到该分段包含哪些 TTS 单词
   - 使用第一个单词的开始时间作为字幕开始时间
   - 使用最后一个单词的结束时间作为字幕结束时间

### 关键代码

```typescript
function mapTimingToSegments(
  wordEntries: SubtitleEntry[],  // TTS单词时间戳
  segments: string[],            // 分段后的文本
  isEnglish: boolean
): SubtitleEntry[] {
  const result: SubtitleEntry[] = [];

  // 1. 构建单词列表（带时间）
  const allWords = wordEntries.flatMap(entry => {
    const words = entry.text.split(/\s+/).filter(w => w.length > 0);
    const duration = entry.end - entry.start;
    const wordDuration = duration / Math.max(words.length, 1);
    let offset = entry.start;
    return words.map(w => ({
      word: w.replace(/[^\w'’-]/g, ''),
      start: offset,
      end: offset + wordDuration
    }));
  });

  // 2. 逐段映射时间
  let wordIndex = 0;
  for (const segment of segments) {
    const segmentLength = isEnglish
      ? segment.split(/\s+/).filter(w => w.length > 0).length
      : segment.replace(/[，。、；：！？\s]/g, '').length;

    // 找到对应单词的开始时间
    const startTime = allWords[wordIndex]?.start || 0;

    // 跳过 segmentLength 个单词来确定结束时间
    let lastWordEnd = startTime;
    for (let i = 0; i < segmentLength && wordIndex < allWords.length; i++) {
      lastWordEnd = allWords[wordIndex].end;
      wordIndex++;
    }

    result.push({
      start: startTime,
      end: lastWordEnd,
      text: segment
    });
  }

  return result;
}
```

## 5. 中文换行处理

ASS 字幕中的 `\N` 是内联换行符，用于在单条字幕内换行。

中文没有空格，长文本不会自动换行，需要手动插入 `\N`：

```typescript
function wrapChineseText(text: string, maxCharsPerLine: number = 18): string {
  if (text.length <= maxCharsPerLine) return text;

  // 找逗号等标点作为换行点
  const mid = Math.floor(text.length / 2);
  const commaMatch = text.substring(0, mid).match(/[,，、；：][^,，、；：]*$/);

  if (commaMatch && commaMatch.index !== undefined) {
    const breakIdx = commaMatch.index + 1;
    return text.substring(0, breakIdx) + '\\N' + text.substring(breakIdx);
  }

  // 否则在中点换行
  return text.substring(0, mid) + '\\N' + text.substring(mid);
}
```

## 6. FFmpeg 烧录字幕

### 命令

```bash
ffmpeg -y \
  -loop 1 -framerate 25 -i "slide.png" \
  -i "audio.mp3" \
  -vf "ass='subtitle.ass'" \
  -c:v libx264 -tune stillimage \
  -c:a aac -b:a 192k \
  -t 30 \
  "output.mp4"
```

### 参数说明

| 参数 | 含义 |
|------|------|
| -loop 1 -i slide.png | 循环图片作为视频帧 |
| -framerate 25 | 25帧/秒 |
| -i audio.mp3 | 音频文件 |
| -vf "ass='subtitle.ass'" | 用libass渲染ASS字幕 |
| -c:v libx264 | H.264视频编码 |
| -t 30 | 视频时长30秒 |

### 渲染原理

1. FFmpeg 的 `libass` 库读取 ASS 文件
2. 根据 PlayResX/PlayResY 确定坐标系
3. 在每个时间点检查字幕显示
4. 使用 Style 中定义的字体、颜色、大小渲染
5. 将渲染好的图像叠加到视频帧上

## 7. 同步原理

字幕与音频严格同步的关键在于：**使用 TTS 引擎自身产生的时间戳来标记字幕时间**。

```
Edge TTS 生成音频
     ↓
同时输出 WordBoundary 事件（每个单词的时间戳）
     ↓
用这些真实的时间戳来设置字幕时间
     ↓
因为字幕时间和音频时间是同一来源，所以严格同步
```

## 8. 项目实现

### 关键文件

| 文件 | 职责 |
|------|------|
| src/services/subtitleGenerator.ts | 字幕生成核心逻辑 |
| src/services/tts.ts | TTS音频生成 |

### 生成流程

```typescript
async function generateSubtitles(
  narrationScript: string,
  audioPath: string,
  outputDir: string,
  sectionId: number
): Promise<string> {
  // 1. 获取TTS单词时间戳
  const wordEntries = await runEdgeTTSWordBoundary(narrationScript);

  // 2. 检测语言
  const isEnglish = detectLanguage(narrationScript);

  // 3. 文本分段
  const segments = splitIntoSegments(narrationScript, isEnglish);

  // 4. 时间映射
  const timedSegments = mapTimingToSegments(wordEntries, segments, isEnglish);

  // 5. 生成ASS
  const assContent = generateASS(timedSegments, isEnglish);

  // 6. 写入文件
  await writeFile(outputPath, assContent, 'utf-8');

  return outputPath;
}
```

## 9. 常见问题

### 字幕不同步

**原因**：时间映射算法有误

**解决**：使用位置-based 映射，而非文本匹配

### 中文不换行

**原因**：ASS 需要 `\N` 手动换行

**解决**：在 `generateASS()` 中调用 `wrapChineseText()` 处理

### 字体显示异常

**原因**：系统未安装指定字体

**解决**：确保视频容器中嵌入字体，或使用系统通用字体
