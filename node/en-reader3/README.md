# EN-READER3

AI-powered English article video generator with Chinese narration.

## Features

- Automatic article segmentation
- AI-generated teaching content (vocabulary, phrases, grammar, context)
- Chinese narration with TTS (Edge TTS or ByteDance)
- **Accurate subtitle timing using faster-whisper ASR** - word-level timestamps ensure subtitles sync with speech
- MP4 video generation with subtitles (9:16 vertical format)

## Prerequisites

- Node.js 18+
- FFmpeg (install via `brew install ffmpeg`)
- Python 3 with faster-whisper: `pip install faster-whisper`
- Edge TTS: `pip install edge-tts` (for default TTS provider)

## Installation

```bash
pnpm install
```

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
# LLM Configuration (required)
LLM_BASE_URL=https://api.minimaxi.com/anthropic
LLM_API_KEY=your-api-key
LLM_MODEL=MiniMax-M2.7

# TTS Provider (default: edge)
TTS_PROVIDER=edge  # or 'bytedance'

# ByteDance TTS (if using bytedance)
BYTEDANCE_APP_ID=your-app-id
BYTEDANCE_ACCESS_KEY=your-access-key
BYTEDANCE_RESOURCE_ID=your-resource-id
```

## Usage

### CLI

```bash
# Generate video from article (full pipeline)
pnpm exec tsx src/index.ts --input ./input/fairy-tale.txt --output ./output/video.mp4
```

### Phased Execution

For long articles or distributed processing, run pipeline phases separately:

```bash
# Phase 1: Segment article only
pnpm exec tsx src/index.ts --input ./input/fairy-tale.txt --phase 1

# Phase 2: Generate AI scripts (requires phase 1 output)
pnpm exec tsx src/index.ts --input ./input/fairy-tale.txt --phase 2

# Phase 3: Generate slides & audio (requires phase 2 output)
pnpm exec tsx src/index.ts --input ./input/fairy-tale.txt --phase 3

# Phase 4: Generate video segments with subtitles (requires phase 3 output)
pnpm exec tsx src/index.ts --input ./input/fairy-tale.txt --phase 4

# Phase 5: Concatenate all segments into final video (requires phase 4 output)
pnpm exec tsx src/index.ts --input ./input/fairy-tale.txt --phase 5

# Full pipeline (default - runs all phases)
pnpm exec tsx src/index.ts --input ./input/fairy-tale.txt --output ./output/video.mp4
```

Each phase reads from and writes to `output/segments/`:
- Phase 1: Creates segment data files
- Phase 2: Adds AI-generated scripts to data files
- Phase 3: Adds slides and audio files
- Phase 4: Generates video segments and subtitle files
- Phase 5: Concatenates segments into final video

### Web API

```bash
# Start server
pnpm exec tsx src/server.ts

# Generate video
curl -X POST http://localhost:3000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"articleText": "Your English article here...", "title": "Article Title"}'

# Check status
curl http://localhost:3000/api/status/{jobId}

# Download video
curl http://localhost:3000/api/download/{jobId} -o output.mp4
```

## How Subtitle Timing Works

1. **TTS generates audio** from the narration script
2. **faster-whisper ASR** transcribes the audio and returns word-level timestamps
3. **Subtitles are generated** from the ASR transcript with proper timing
4. **Line splitting** groups words into readable chunks (~20-25 characters) without splitting words

This ensures subtitles are synchronized with actual speech - no more estimated timing based on character count.

## Output

**Video:** MP4 in 9:16 vertical format (1080x1920) with:
- Slide showing key teaching points (vocabulary, grammar, context)
- Chinese narration audio (TTS reads the full script)
- Synchronized subtitles showing what is actually spoken

**JSON data files:** Each section has a `data.json` containing:
- Original English text
- Key vocabulary, grammar points, context explanation
- Full narration script (used for TTS)

**Subtitle files:** `section-X.srt` with ASR-verified timing
