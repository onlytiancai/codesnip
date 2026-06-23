## Per-file results

| file | wav2aac wall | ffmpeg wall | speedup | wav2aac RSS | ffmpeg RSS | RSS ratio | out size (w2a) | out size (ff) |
|------|-------------:|------------:|--------:|------------:|-----------:|----------:|--------------:|--------------:|
| *ch2*1s*.wav | 5.5 ms | 0.017 s | 3.09x | 5.4 MB | 10.4 MB | 0.52x | 0 B | 0 B |
| sr48000_ch1_bd16_10s.wav | 0.028 s | 0.059 s | 2.08x | 11.6 MB | 21.0 MB | 0.55x | 160.0 KB | 159.9 KB |
| sr48000_ch1_bd16_1s.wav | 0.013 s | 0.035 s | 2.56x | 11.3 MB | 20.2 MB | 0.56x | 14.5 KB | 17.6 KB |
| sr48000_ch1_bd16_60s.wav | 0.111 s | 0.170 s | 1.54x | 13.3 MB | 21.2 MB | 0.63x | 946.0 KB | 950.4 KB |
| sr48000_ch1_bd24_10s.wav | 0.029 s | 0.050 s | 1.71x | 11.6 MB | 21.1 MB | 0.55x | 160.0 KB | 159.9 KB |
| sr48000_ch1_bd24_1s.wav | 0.014 s | 0.028 s | 2.01x | 11.3 MB | 20.5 MB | 0.55x | 14.5 KB | 17.6 KB |
| sr48000_ch1_bd24_60s.wav | 0.112 s | 0.163 s | 1.46x | 13.3 MB | 21.5 MB | 0.62x | 946.0 KB | 950.4 KB |
| sr48000_ch2_bd16_10s.wav | 0.048 s | 0.105 s | 2.18x | 11.9 MB | 21.9 MB | 0.54x | 98.1 KB | 159.8 KB |
| sr48000_ch2_bd16_1s.wav | 0.015 s | 0.040 s | 2.57x | 11.6 MB | 20.9 MB | 0.56x | 7.5 KB | 17.5 KB |
| sr48000_ch2_bd16_60s.wav | 0.252 s | 0.402 s | 1.59x | 13.2 MB | 22.1 MB | 0.60x | 725.9 KB | 950.3 KB |
| sr48000_ch2_bd24_10s.wav | 0.050 s | 0.088 s | 1.75x | 11.9 MB | 22.2 MB | 0.54x | 96.4 KB | 159.8 KB |
| sr48000_ch2_bd24_1s.wav | 0.015 s | 0.032 s | 2.08x | 11.6 MB | 21.0 MB | 0.55x | 6.9 KB | 17.5 KB |
| sr48000_ch2_bd24_60s.wav | 0.269 s | 0.387 s | 1.44x | 13.3 MB | 22.6 MB | 0.59x | 747.0 KB | 950.3 KB |
| sr96000_ch1_bd16_10s.wav | 0.037 s | 0.069 s | 1.87x | 11.6 MB | 21.9 MB | 0.53x | 160.0 KB | 159.9 KB |
| sr96000_ch1_bd16_1s.wav | 0.016 s | 0.037 s | 2.33x | 11.3 MB | 21.0 MB | 0.54x | 14.5 KB | 17.6 KB |
| sr96000_ch1_bd16_60s.wav | 0.166 s | 0.183 s | 1.11x | 13.4 MB | 22.2 MB | 0.60x | 946.0 KB | 950.4 KB |
| sr96000_ch1_bd24_10s.wav | 0.039 s | 0.051 s | 1.31x | 11.6 MB | 22.1 MB | 0.53x | 160.0 KB | 159.9 KB |
| sr96000_ch1_bd24_1s.wav | 0.015 s | 0.029 s | 1.93x | 11.3 MB | 21.0 MB | 0.54x | 14.5 KB | 17.6 KB |
| sr96000_ch1_bd24_60s.wav | 0.177 s | 0.167 s | 0.94x | 13.3 MB | 22.5 MB | 0.59x | 946.0 KB | 950.4 KB |
| sr96000_ch2_bd16_10s.wav | 0.066 s | 0.141 s | 2.14x | 11.9 MB | 24.4 MB | 0.49x | 98.1 KB | 159.8 KB |
| sr96000_ch2_bd16_1s.wav | 0.023 s | 0.054 s | 2.37x | 11.6 MB | 21.9 MB | 0.53x | 7.5 KB | 17.5 KB |
| sr96000_ch2_bd16_60s.wav | 0.313 s | 0.456 s | 1.46x | 13.3 MB | 24.8 MB | 0.53x | 725.9 KB | 950.3 KB |
| sr96000_ch2_bd24_10s.wav | 0.064 s | 0.096 s | 1.49x | 11.9 MB | 24.7 MB | 0.48x | 96.4 KB | 159.8 KB |
| sr96000_ch2_bd24_1s.wav | 0.019 s | 0.035 s | 1.85x | 11.6 MB | 21.8 MB | 0.53x | 6.9 KB | 17.5 KB |
| sr96000_ch2_bd24_60s.wav | 0.339 s | 0.416 s | 1.23x | 13.3 MB | 25.2 MB | 0.53x | 747.0 KB | 950.3 KB |

## Summary across 25 files

- **Median speedup (wall time, ffmpeg ÷ wav2aac)**: 1.85x
- **Min speedup**: 0.94x, **Max speedup**: 3.09x
- **Median RSS ratio (wav2aac ÷ ffmpeg)**: 0.54x (lower = wav2aac uses less memory)

## Speedup by file-size bucket

| bucket | median speedup |
|--------|---------------:|
| medium (60–600s) | 1.46x |
| small (5–60s) | 1.87x |
| tiny (<5s) | 2.33x |
| unknown | 3.09x |
