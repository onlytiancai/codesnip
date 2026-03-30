# Compile FFmpeg with libass Support

## Problem

The Homebrew pre-built FFmpeg does not include the `ass` filter (libass). When running:
```bash
ffmpeg -vf "ass='subtitle.ass'" ...
```

You get error:
```
No option name near 'subtitle.ass'
```

## Solution

### Step 1: Download FFmpeg Source

```bash
cd /tmp
curl -LO https://ffmpeg.org/releases/ffmpeg-8.1.tar.xz
tar xf ffmpeg-8.1.tar.xz
cd ffmpeg-8.1
```

### Step 2: Install Dependencies

```bash
brew install pkg-config yasm nasm freetype fribidi fontconfig libass
```

### Step 3: Configure with libass

```bash
./configure --prefix=/opt/homebrew \
  --enable-libass \
  --enable-gpl \
  --enable-version3 \
  --enable-libx264 \
  --enable-libx265 \
  --enable-libvpx \
  --enable-openssl \
  --enable-videotoolbox \
  --enable-audiotoolbox \
  --enable-pthreads \
  --enable-shared
```

Key option: `--enable-libass`

### Step 4: Compile and Install

```bash
make -j$(sysctl -n hw.ncpu)
sudo make install
```

### Step 5: Verify

```bash
ffmpeg -filters 2>/dev/null | grep ass
```

Should show:
```
.. ass               V->V       Render ASS subtitles onto input video using the libass library.
```

Check version:
```bash
ffmpeg -version 2>&1 | grep configuration
```

Should include `--enable-libass`.

## Usage

With libass, you can burn ASS subtitles into video:
```bash
ffmpeg -y \
  -loop 1 -framerate 25 -i slide.png \
  -i audio.mp3 \
  -vf "ass='subtitle.ass'" \
  -c:v libx264 -tune stillimage \
  -c:a aac -b:a 192k \
  -shortest \
  output.mp4
```

## Alternative: Use PIL-based Rendering

If FFmpeg cannot be compiled with libass, use PIL-based frame rendering (see `scripts/generate-karaoke-video.py`):

1. Parse words.txt for word timings
2. Render each frame with PIL (karaoke highlight effect)
3. Encode video with FFmpeg image input
