use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::error::Error;
use std::fs::File;
use std::path::PathBuf;

use clap::Parser;
use csv::{ReaderBuilder, StringRecord, WriterBuilder};
use tempfile::TempDir;

const MAX_OPEN_FILES: usize = 64;

#[derive(Parser, Debug)]
struct Args {
    input: PathBuf,
    output: PathBuf,

    #[arg(long)]
    chunk_size: usize,

    #[arg(long)]
    sort_col: usize,

    #[arg(long, default_value_t = false)]
    dedup: bool,
}

/* ---------- heap item ---------- */

#[derive(Debug)]
struct HeapItem {
    key: String,
    chunk_idx: usize,
    record: StringRecord,
}

impl Eq for HeapItem {}

impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        other.key.cmp(&self.key) // min-heap
    }
}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/* ---------- main ---------- */

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let tempdir = TempDir::new()?;
    println!("临时目录: {:?}", tempdir.path());

    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .from_path(&args.input)?;

    let headers = rdr.headers()?.clone();

    println!("开始分块排序…");

    let mut chunks: Vec<PathBuf> = Vec::new();
    let mut buffer = Vec::with_capacity(args.chunk_size);

    for (i, row) in rdr.records().enumerate() {
        buffer.push(row?);

        if buffer.len() == args.chunk_size {
            let path = tempdir
                .path()
                .join(format!("chunk_{}.csv", chunks.len()));
            write_sorted_chunk(&buffer, &headers, args.sort_col, &path)?;
            chunks.push(path);
            buffer.clear();
            println!("已完成分块 {}", chunks.len());
        }

        if (i + 1) % 1_000_000 == 0 {
            println!("已读取 {} 行", i + 1);
        }
    }

    if !buffer.is_empty() {
        let path = tempdir
            .path()
            .join(format!("chunk_{}.csv", chunks.len()));
        write_sorted_chunk(&buffer, &headers, args.sort_col, &path)?;
        chunks.push(path);
        println!("已完成分块 {}", chunks.len());
    }

    println!("分块完成，共 {} 个 chunk", chunks.len());

    let final_path = merge_in_rounds(
        chunks,
        &headers,
        args.sort_col,
        args.dedup,
        tempdir.path(),
    )?;

    std::fs::copy(&final_path, &args.output)?;
    println!("排序完成，输出写入 {:?}", args.output);

    Ok(())
}

/* ---------- write chunk ---------- */

fn write_sorted_chunk(
    records: &[StringRecord],
    headers: &StringRecord,
    sort_col: usize,
    path: &PathBuf,
) -> Result<(), Box<dyn Error>> {
    let mut data = records.to_vec();
    data.sort_by(|a, b| a[sort_col].cmp(&b[sort_col]));

    let file = File::create(path)?;
    let mut wtr = WriterBuilder::new().has_headers(true).from_writer(file);

    wtr.write_record(headers)?;
    for r in data {
        wtr.write_record(&r)?;
    }
    wtr.flush()?;
    Ok(())
}

/* ---------- multi-pass merge ---------- */

fn merge_in_rounds(
    mut chunks: Vec<PathBuf>,
    headers: &StringRecord,
    sort_col: usize,
    dedup: bool,
    temp_root: &std::path::Path,
) -> Result<PathBuf, Box<dyn Error>> {
    let mut round = 0;

    while chunks.len() > 1 {
        round += 1;
        println!(
            "开始第 {} 轮归并，chunk 数 = {}",
            round,
            chunks.len()
        );

        let mut next = Vec::new();

        for (i, group) in chunks.chunks(MAX_OPEN_FILES).enumerate() {
            let out = temp_root.join(format!("merge_r{}_{}.csv", round, i));
            merge_group(group, headers, sort_col, dedup, &out)?;
            next.push(out);
        }

        chunks = next;
    }

    Ok(chunks.remove(0))
}

/* ---------- merge one group ---------- */

fn merge_group(
    group: &[PathBuf],
    headers: &StringRecord,
    sort_col: usize,
    dedup: bool,
    out_path: &PathBuf,
) -> Result<(), Box<dyn Error>> {
    let mut readers = Vec::new();

    for p in group {
        let f = File::open(p)?;
        readers.push(ReaderBuilder::new().has_headers(true).from_reader(f));
    }

    let mut heap = BinaryHeap::<HeapItem>::new();

    for (idx, rdr) in readers.iter_mut().enumerate() {
        if let Some(r) = rdr.records().next() {
            let rec = r?;
            heap.push(HeapItem {
                key: rec[sort_col].to_string(),
                chunk_idx: idx,
                record: rec,
            });
        }
    }

    let out_file = File::create(out_path)?;
    let mut wtr = WriterBuilder::new().has_headers(true).from_writer(out_file);
    wtr.write_record(headers)?;

    let mut last_key: Option<String> = None;

    while let Some(item) = heap.pop() {
        let emit = if dedup {
            last_key.as_deref() != Some(&item.key)
        } else {
            true
        };

        if emit {
            wtr.write_record(&item.record)?;
            last_key = Some(item.key.clone());
        }

        let idx = item.chunk_idx;
        if let Some(r) = readers[idx].records().next() {
            let rec = r?;
            heap.push(HeapItem {
                key: rec[sort_col].to_string(),
                chunk_idx: idx,
                record: rec,
            });
        }
    }

    wtr.flush()?;
    Ok(())
}
