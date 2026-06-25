#!/usr/bin/env python3
"""
analyze_results.py — Read bench/results.csv and emit a Markdown summary
table with speedup ratios and RSS comparison, grouped by file size bucket.

Usage:
    bench/analyze_results.py [path/to/results.csv]  # default: bench/results.csv
"""

import csv
import sys
import os
from collections import defaultdict

PYTHON = "/Users/huhao/.pyenv/versions/3.11.9/bin/python"

def load(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def fmt_time(s):
    try:
        s = float(s)
    except (TypeError, ValueError):
        return "N/A"
    if s < 0.01:
        return f"{s*1000:.1f} ms"
    return f"{s:.3f} s"

def fmt_size(b):
    try:
        b = int(b)
    except (TypeError, ValueError):
        return "N/A"
    if b > 1024*1024:
        return f"{b/1024/1024:.1f} MB"
    if b > 1024:
        return f"{b/1024:.1f} KB"
    return f"{b} B"

def fmt_mem(b):
    try:
        b = int(b)
    except (TypeError, ValueError):
        return "N/A"
    return f"{b/1024/1024:.1f} MB"

def speedup(w_wav2aac, w_ffmpeg):
    try:
        a, b = float(w_wav2aac), float(w_ffmpeg)
        if a > 0:
            return b / a
    except (TypeError, ValueError):
        pass
    return None

def main():
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "results.csv")
    if not os.path.exists(src):
        print(f"No results file at {src}", file=sys.stderr)
        sys.exit(1)
    rows = load(src)
    if not rows:
        print("Empty results", file=sys.stderr)
        sys.exit(1)

    # Index by (file, tool)
    by_key = {(r["file"], r["tool"]): r for r in rows}
    files = sorted({r["file"] for r in rows})

    # Per-file comparison
    print("## Per-file results\n")
    print("| file | wav2aac wall | ffmpeg wall | speedup | wav2aac RSS | ffmpeg RSS | RSS ratio | out size (w2a) | out size (ff) |")
    print("|------|-------------:|------------:|--------:|------------:|-----------:|----------:|--------------:|--------------:|")
    speedups = []
    rss_ratios = []
    for f in files:
        a = by_key.get((f, "wav2aac"))
        b = by_key.get((f, "ffmpeg"))
        if not a or not b:
            continue
        sp = speedup(a["wall_s"], b["wall_s"])
        # peak_rss_kb column is actually bytes (macOS /usr/bin/time -l reports bytes)
        rss_a, rss_b = int(a["peak_rss_kb"] or 0), int(b["peak_rss_kb"] or 0)
        rss_ratio = rss_a / rss_b if rss_b > 0 else None
        if sp is not None: speedups.append(sp)
        if rss_ratio is not None: rss_ratios.append(rss_ratio)
        sp_str = f"{sp:.2f}x" if sp else "n/a"
        rs_str = f"{rss_ratio:.2f}x" if rss_ratio else "n/a"
        print(f"| {f} | {fmt_time(a['wall_s'])} | {fmt_time(b['wall_s'])} | {sp_str} | "
              f"{fmt_size(rss_a)} | {fmt_size(rss_b)} | {rs_str} | "
              f"{fmt_size(a['out_bytes'])} | {fmt_size(b['out_bytes'])} |")

    # Aggregate
    if speedups:
        speedups.sort()
        median = speedups[len(speedups)//2]
        print(f"\n## Summary across {len(speedups)} files\n")
        print(f"- **Median speedup (wall time, ffmpeg ÷ wav2aac)**: {median:.2f}x")
        print(f"- **Min speedup**: {min(speedups):.2f}x, **Max speedup**: {max(speedups):.2f}x")
    if rss_ratios:
        rss_ratios.sort()
        print(f"- **Median RSS ratio (wav2aac ÷ ffmpeg)**: {rss_ratios[len(rss_ratios)//2]:.2f}x "
              f"(lower = wav2aac uses less memory)")

    # Bucket by audio duration
    by_dur = defaultdict(list)
    for f in files:
        a = by_key.get((f, "wav2aac"))
        b = by_key.get((f, "ffmpeg"))
        if not a or not b: continue
        dur = float(a.get("audio_s") or 0)
        if dur <= 0: bucket = "unknown"
        elif dur < 5:   bucket = "tiny (<5s)"
        elif dur < 60:  bucket = "small (5–60s)"
        elif dur < 600: bucket = "medium (60–600s)"
        else:           bucket = "large (>600s)"
        by_dur[bucket].append(speedup(a["wall_s"], b["wall_s"]))
    print("\n## Speedup by file-size bucket\n")
    print("| bucket | median speedup |")
    print("|--------|---------------:|")
    for b in sorted(by_dur):
        vals = sorted([v for v in by_dur[b] if v is not None])
        if not vals: continue
        med = vals[len(vals)//2]
        print(f"| {b} | {med:.2f}x |")

if __name__ == "__main__":
    main()