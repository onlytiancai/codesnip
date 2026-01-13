运行

    cargo run --release -- \
    input.csv output.csv \
    --chunk-size 500000 \
    --sort-col 2 \
    --dedup