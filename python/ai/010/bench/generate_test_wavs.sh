#!/usr/bin/env bash
#
# generate_test_wavs.sh — Build a test matrix of WAV files covering
# common combinations of sample rate, channels, bit depth, and duration.
# Output goes to bench/wavs/ alongside this script.
#
# Test matrix:
#   sample rates: 22050, 44100, 48000, 96000 Hz
#   channels:     1, 2
#   bit depths:   16, 24
#   durations:    1s, 10s, 60s, 600s
#   content:      440 Hz sine (constant) — exercises AAC encoder predictably
#
# We also keep a tiny 1s clip and a large 1GB-ish clip for stress testing.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/wavs"
mkdir -p "$OUT"

# 1s sine generation is fast; longer ones use larger duration params
gen_wav() {
    local sr=$1
    local ch=$2
    local bd=$3
    local dur=$4
    local name="$OUT/sr${sr}_ch${ch}_bd${bd}_${dur}s.wav"
    if [[ -f "$name" ]]; then
        echo "skip: $name"
        return
    fi
    local ffmpeg_bd
    case "$bd" in
        16) ffmpeg_bd="pcm_s16le" ;;
        24) ffmpeg_bd="pcm_s24le" ;;
        32) ffmpeg_bd="pcm_f32le" ;;
        *)  echo "unsupported bit depth $bd"; return 1 ;;
    esac
    ffmpeg -y -hide_banner -loglevel error \
        -f lavfi -i "sine=frequency=440:duration=${dur}:sample_rate=${sr}" \
        -ac "$ch" -c:a "$ffmpeg_bd" "$name"
    local sz=$(stat -f%z "$name")
    echo "wrote: $name (${sz} bytes)"
}

# Small matrix for normal benchmark
for sr in 44100 48000 96000; do
    for ch in 1 2; do
        for bd in 16 24; do
            for dur in 1 10 60 600; do
                gen_wav "$sr" "$ch" "$bd" "$dur"
            done
        done
    done
done

echo "==="
echo "Generated $(ls "$OUT" | wc -l | tr -d ' ') WAV files in $OUT"
du -sh "$OUT"