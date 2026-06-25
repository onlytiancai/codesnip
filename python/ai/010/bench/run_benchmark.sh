#!/usr/bin/env bash
#
# run_benchmark.sh — Compare wav2aac vs ffmpeg-aac_at across the test
# matrix. Each (file, tool) pair is run `RUNS` times; we report the median
# of wall clock + user CPU + max RSS + output size.
#
# Output: bench/results.csv with columns
#   tool, file, wall_s, cpu_user_s, cpu_sys_s, peak_rss_kb, out_bytes,
#   audio_s, realtime_factor

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
WAV2AAC="$ROOT/build/wav2aac"
WAV_DIR="$HERE/wavs"
RESULTS="$HERE/results.csv"
RUNS=${RUNS:-5}
FFMPEG_BIN=${FFMPEG_BIN:-/opt/homebrew/bin/ffmpeg}

if [[ ! -x "$WAV2AAC" ]]; then
    echo "wav2aac binary not built. Run 'make' first."
    exit 1
fi
if [[ ! -d "$WAV_DIR" || -z "$(ls -A "$WAV_DIR" 2>/dev/null)" ]]; then
    echo "WAV directory empty. Run ./generate_test_wavs.sh first."
    exit 1
fi

# macOS /usr/bin/time lacks -o, so we use a custom runner that captures
# wall, CPU, and RSS by sampling /usr/bin/time -l stderr to a file.
#
# $1 = tool (wav2aac|ffmpeg)
# $2 = input wav
# $3 = output m4a
# writes to /tmp/w2a_metrics.txt in INI-like format
run_one() {
    local tool=$1
    local in=$2
    local out=$3
    local t0 t1 wall cpu_u cpu_s rss out_bytes

    t0=$(date +%s.%N)
    if [[ "$tool" == "wav2aac" ]]; then
        /usr/bin/time -l "$WAV2AAC" "$in" "$out" 2>/tmp/w2a_time.txt >/dev/null
    else
        /usr/bin/time -l "$FFMPEG_BIN" -y -hide_banner -loglevel error \
            -i "$in" -c:a aac_at -b:a 128k -ar 48000 -vn "$out" 2>/tmp/w2a_time.txt >/dev/null
    fi
    t1=$(date +%s.%N)
    wall=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.4f", b-a}')

    # macOS /usr/bin/time -l puts "real X user Y sys Z" on a single line first.
    # Extract user and sys by sed: pattern "<digits>.<digits> user" / "... sys"
    cpu_u=$(sed -n 's/.*[[:space:]]\([0-9.]*\) user.*/\1/p' /tmp/w2a_time.txt)
    cpu_s=$(sed -n 's/.*[[:space:]]\([0-9.]*\) sys.*/\1/p' /tmp/w2a_time.txt)
    rss=$(awk '/maximum resident set size/{print $1}' /tmp/w2a_time.txt)
    out_bytes=$(stat -f%z "$out" 2>/dev/null || echo 0)

    echo "tool=$tool"
    echo "wall=$wall"
    echo "cpu_u=${cpu_u:-0}"
    echo "cpu_s=${cpu_s:-0}"
    echo "rss=${rss:-0}"
    echo "out=$out_bytes"
}

# Median helper: pick the floor(N/2)+1'th element after sort
median_n() {
    awk -v n="$RUNS" '{print $1}' | sort -n | awk -v n="$n" 'NR==int((n+1)/2){print;exit}'
}

# Run RUNS trials for one (tool, wav); write median to results.csv
run_median() {
    local tool=$1
    local wav=$2
    local base=$(basename "$wav")
    local dur
    dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$wav" 2>/dev/null || echo 0)

    # Warmup
    if [[ "$tool" == "wav2aac" ]]; then
        "$WAV2AAC" "$wav" "$HERE/out_${tool}_${base%.wav}_warmup.m4a" >/dev/null 2>&1 || true
    else
        "$FFMPEG_BIN" -y -hide_banner -loglevel error -i "$wav" \
            -c:a aac_at -b:a 128k -ar 48000 -vn \
            "$HERE/out_${tool}_${base%.wav}_warmup.m4a" >/dev/null 2>&1 || true
    fi

    local i walls=() cpus=() syss=() rsss=() outs=()
    for i in $(seq 1 "$RUNS"); do
        local out="$HERE/out_${tool}_${base%.wav}_run${i}.m4a"
        if [[ "$tool" == "wav2aac" ]]; then
            t0=$(date +%s.%N)
            /usr/bin/time -l "$WAV2AAC" "$wav" "$out" 2>/tmp/w2a_t.txt >/dev/null || true
            t1=$(date +%s.%N)
        else
            t0=$(date +%s.%N)
            /usr/bin/time -l "$FFMPEG_BIN" -y -hide_banner -loglevel error \
                -i "$wav" -c:a aac_at -b:a 128k -ar 48000 -vn "$out" 2>/tmp/w2a_t.txt >/dev/null || true
            t1=$(date +%s.%N)
        fi
        walls+=("$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.4f", b-a}')")
        cpus+=("$(sed -n 's/.*[[:space:]]\([0-9.]*\) user.*/\1/p' /tmp/w2a_t.txt)")
        syss+=("$(sed -n 's/.*[[:space:]]\([0-9.]*\) sys.*/\1/p' /tmp/w2a_t.txt)")
        rsss+=("$(awk '/maximum resident set size/{print $1}' /tmp/w2a_t.txt)")
        outs+=("$(stat -f%z "$out" 2>/dev/null || echo 0)")
    done

    local mwall mcpu msys mrss mout rtf
    mwall=$(printf '%s\n' "${walls[@]}" | sort -n | awk -v n="$RUNS" 'NR==int((n+1)/2){print;exit}')
    mcpu=$(printf  '%s\n' "${cpus[@]}"  | sort -n | awk -v n="$RUNS" 'NR==int((n+1)/2){print;exit}')
    msys=$(printf  '%s\n' "${syss[@]}"  | sort -n | awk -v n="$RUNS" 'NR==int((n+1)/2){print;exit}')
    mrss=$(printf  '%s\n' "${rsss[@]}"  | sort -n | awk -v n="$RUNS" 'NR==int((n+1)/2){print;exit}')
    mout=$(printf  '%s\n' "${outs[@]}"  | sort -n | awk -v n="$RUNS" 'NR==int((n+1)/2){print;exit}')
    rtf=$(awk -v w="$mwall" -v d="$dur" 'BEGIN{if(d>0){printf "%.4f", w/d}else{print "inf"}}')

    echo "$tool,$base,${mwall},${mcpu},${msys},${mrss},${mout},${dur},${rtf}" >> "$RESULTS"
}

# Main loop
shopt -s nullglob
wavs=( "$WAV_DIR"/*.wav )
echo "tool,file,wall_s,cpu_user_s,cpu_sys_s,peak_rss_kb,out_bytes,audio_s,realtime_factor" > "$RESULTS"
echo "Running benchmark: ${#wavs[@]} files x 2 tools x $RUNS runs (median)"

for wav in "${wavs[@]}"; do
    base=$(basename "$wav")
    printf '  %-40s  wav2aac...' "$base"
    run_median wav2aac "$wav"
    printf ' ffmpeg...'
    run_median ffmpeg    "$wav"
    echo " done"
done

echo "==="
echo "Results: $RESULTS ($(wc -l < "$RESULTS") rows including header)"
echo "See bench/analyze_results.py for summary tables."