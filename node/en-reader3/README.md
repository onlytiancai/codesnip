# EN-READER3

AI-powered English article video generator with Chinese narration.

## Features

- Automatic article segmentation
- AI-generated teaching content (vocabulary, phrases, grammar, context)
- Chinese narration with TTS
- MP4 video generation with subtitles (9:16 vertical format)

## Prerequisites

- Node.js 18+
- FFmpeg (install via `brew install ffmpeg`)
- Python 3 (for Edge TTS): `pip install edge-tts`

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
# Generate video from article
pnpm exec tsx src/index.ts --input ./input/fairy-tale.txt --output ./output/video.mp4
```

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

## Output

**Video:** MP4 in 9:16 vertical format (1080x1920) with:
- Slide showing key teaching points (vocabulary, grammar, context)
- Chinese narration audio (TTS reads the full script)
- Synchronized Chinese subtitles (showing narration script as it's spoken)

**JSON data files:** Each section has a `data.json` containing:
- Original English text
- Key vocabulary, grammar points, context explanation
- Full narration script (used for TTS and subtitles)
