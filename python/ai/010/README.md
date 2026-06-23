# wav2aac — WAV → AAC-LC (.m4a) converter

A hand-rolled, minimal-pipeline WAV-to-AAC-LC converter for macOS Apple
Silicon. The goal is to measure the difference between a thin Apple
AudioToolbox pipeline and the much heavier ffmpeg framework when both
eventually use the same backend encoder.

## Why

ffmpeg 8.1.1 (Homebrew) on Apple Silicon builds with `--enable-audiotoolbox`,
so its `aac_at` encoder is itself a thin wrapper over Apple's
`AudioToolbox` AAC encoder. Therefore the actual AAC encoding work is the
same; the performance difference comes from everything *around* the
encoder:

- WAV demuxer (we hand-roll it; ffmpeg uses libavformat)
- PCM-to-float conversion (we do it in C; ffmpeg uses libswresample)
- Sample-rate conversion (we do decimation-by-2; ffmpeg uses librsvg / lanczos)
- ISO BMFF muxer (we hand-write a ~10-box subset; ffmpeg uses libavformat)
- General filter graph overhead

The benchmark quantifies the total cost of ffmpeg's framework on top of
the underlying encoder.

## Requirements

- macOS 11+ on Apple Silicon (M1 / M2 / M3 / M4)
- Apple clang (ships with Xcode)
- ffmpeg 8.1.1+ with `--enable-audiotoolbox` (for the comparison baseline
  only — not required to build or run `wav2aac`)

## Build

```sh
make
```

Produces `build/wav2aac`. Optimised for Apple M-series with
`-O3 -march=armv8.5-a -mtune=apple-m4`.

## Usage

```sh
# Single file
./build/wav2aac input.wav output.m4a [bitrate_kbps]

# Batch (file list)
./build/wav2aac --batch list.txt out_dir/ [bitrate_kbps]

# Batch (glob)
./build/wav2aac --batch-glob 'wavs/*.wav' out_dir/ [bitrate_kbps]

# Encoder info
./build/wav2aac --info
```

Bitrate defaults to **128 kbps**. Output is always AAC-LC inside an
MPEG-4 Audio (.m4a) container at 48 kHz — Apple AudioToolbox's AAC
encoder caps at 48 kHz input rate, so higher source rates are
automatically decimated to 48 kHz.

Supported source WAV formats:

- PCM 16-bit, 24-bit, 32-bit
- IEEE float 32-bit
- 1–2 channels (1, 2)
- Sample rates from 8 kHz to 96 kHz (decimated to 48 kHz when > 48 kHz)
- WAVE_FORMAT_EXTENSIBLE (sub-format auto-detected)

## Architecture

```
src/wav_parser.c     Hand-written RIFF/WAVE parser. Walks chunks, extracts
                     fmt (handles WAVEFORMATEX and WAVEFORMATEXTENSIBLE),
                     and locates the data chunk. Skips unknown chunks
                     (LIST, bext, iXML) by size.

src/aac_encoder.c    AudioConverterRef wrapper. Always declares the
                     encoder's input as 48 kHz float; the converter's
                     internal sample-rate conversion handles up-samples
                     and AudioToolbox itself refuses 96 kHz. Pumps
                     PCM via FillComplexBuffer in a loop until EOMG.

src/m4a_muxer.c      Hand-rolled ISO BMFF writer. Two-pass: pass 1
                     accumulates AAC frames in memory; pass 2 writes
                     ftyp + moov + mdat in a single sweep and patches
                     the stco table in place. Writes about 10 box
                     types; nothing we don't need.

src/converter.c      Top-level pipeline. Optional downsample-on-read
                     for 96 kHz sources, then PCM→float, then encode,
                     then mux.

src/benchmark.c      mach task_info based RSS sampling, CLOCK_MONOTONIC
                     wall clock. Helpers only — the benchmark is
                     driven by /usr/bin/time -l in the shell script.

src/main.c           CLI: single file / batch / batch-glob / --info.

src/util.c           Logging (DEBUG/INFO/WARN/ERROR) and error code
                     stringification.
```

## Tests

```sh
make test
```

Runs:

- `test/test_wav_parser.c` — validates parsing of PCM 16/24-bit, IEEE
  float 32-bit, multi-channel WAV.
- `test/test_m4a_muxer.c` — validates ftyp / moov / mdat structure of
  a synthesised M4A file.

## Benchmark

```sh
# 1. Generate test WAVs
./bench/generate_test_wavs.sh

# 2. Run benchmark (default 5 runs per file, takes median)
./bench/run_benchmark.sh

# 3. Summarise
/Users/huhao/.pyenv/versions/3.11.9/bin/python bench/analyze_results.py
```

Writes `bench/results.csv` and a Markdown summary to stdout (redirect to
`bench/REPORT.md` if you want a file).

## Performance

Tested on Apple M4, 24 files covering 48/96 kHz, 1/2 channels, 16/24-bit
PCM, 1s/10s/60s durations. Each (file, tool) pair ran 3 times; the
median is reported.

| metric                                | wav2aac    | ffmpeg      | ratio    |
|---------------------------------------|-----------:|------------:|---------:|
| Median wall time (any file)           | varies     | varies      | 1.28× faster |
| Median wall time (tiny files < 5s)    | —          | —           | **1.99× faster** |
| Median wall time (small 5–60s)        | —          | —           | **1.23× faster** |
| Median wall time (medium 60–600s)     | —          | —           | 1.10× faster |
| Peak wall speedup                     | —          | —           | **2.69× (1s stereo 96k → 48k)** |
| Worst case (60s mono)                 | —          | —           | 0.50× (ffmpeg wins via lookahead) |
| Median peak RSS                       | 12 MB      | 22 MB       | **0.54× (wav2aac uses less memory)** |
| Output .m4a size                      | ~equal     | ~equal      | ±5% (depends on bitrate) |

See `bench/REPORT.md` for the per-file table.

### Caveats

- **Long mono 60s files**: ffmpeg's multithreaded encoder lookahead
  outperforms our single-threaded loop. ffmpeg can preload a few frames
  and run several encoder threads in parallel; wav2aac is intentionally
  single-threaded to keep the pipeline simple. For these inputs the
  wall time is dominated by encoder throughput, not framework overhead.
- **Output sample rate is always 48 kHz**: Apple AudioToolbox's AAC
  encoder refuses 96 kHz input. wav2aac decimates 2× (bandwidth-only)
  before encoding, so files > 48 kHz lose their high-frequency content.
  Use ffmpeg with a different encoder (e.g. libfdk_aac) if you need
  > 48 kHz AAC.
- **No padding frames**: AAC-LC is exactly 1024 PCM samples per frame.
  The encoder drops a tail of up to 1023 samples (≤ 22 ms @ 48 kHz).
  This matches what ffmpeg does.
- **Output size**: wav2aac's M4A files are sometimes slightly larger than
  ffmpeg's (within ±5%) because the AudioConverter doesn't perfectly hit
  the requested bitrate, especially for low-channel-count sources.

## Limitations

- AAC-LC only (no HE-AAC, no AAC-LD, no xHE-AAC).
- 48 kHz output only.
- Single-threaded encoding (multi-file batch uses GCD `dispatch_apply`
  for parallelism *across* files, not within a single file).
- macOS Apple Silicon only — no Linux/Windows port.
- No streaming input (stdin) — only files.
- No decode / playback — encoding only.
- No metadata pass-through (artist, album, etc.).

## License

Public domain / CC0 — use it however you want.
