use std::collections::BinaryHeap;
use std::error::Error;
use std::path::PathBuf;

use clap::Parser;
use csv::{ReaderBuilder, StringRecord, WriterBuilder};
use tempfile::NamedTempFile;

use std::cmp::Ordering;


#[derive(Parser, Debug)]
#[command(author, version, about)]
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
        // BinaryHeap 是最大堆，这里反过来实现，变成最小堆
        other.key.cmp(&self.key)
    }
}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .from_path(&args.input)?;

    let headers = rdr.headers()?.clone();

    let mut chunks: Vec<NamedTempFile> = Vec::new();
    let mut buffer: Vec<StringRecord> = Vec::with_capacity(args.chunk_size);

    println!("开始分块排序...");

    for (i, result) in rdr.records().enumerate() {
        let record = result?;
        buffer.push(record);

        if buffer.len() == args.chunk_size {
            let chunk = write_sorted_chunk(&buffer, &headers, args.sort_col)?;
            chunks.push(chunk);
            buffer.clear();
            println!("已完成分块 {}", chunks.len());
        }

        if (i + 1) % 1_000_000 == 0 {
            println!("已读取 {} 行", i + 1);
        }
    }

    if !buffer.is_empty() {
        let chunk = write_sorted_chunk(&buffer, &headers, args.sort_col)?;
        chunks.push(chunk);
        println!("已完成分块 {}", chunks.len());
    }

    println!("分块完成，共 {} 个块，开始归并排序...", chunks.len());

    merge_chunks(
        chunks,
        &headers,
        &args.output,
        args.sort_col,
        args.dedup,
    )?;

    println!("排序完成，结果已写入 {:?}", args.output);
    Ok(())
}

fn write_sorted_chunk(
    records: &[StringRecord],
    headers: &StringRecord,
    sort_col: usize,
) -> Result<NamedTempFile, Box<dyn Error>> {
    let mut data = records.to_vec();

    data.sort_by(|a, b| a[sort_col].cmp(&b[sort_col]));

    let mut tmp = NamedTempFile::new()?;
    {
        let mut wtr = WriterBuilder::new().has_headers(true).from_writer(&mut tmp);
        wtr.write_record(headers)?;
        for r in data {
            wtr.write_record(&r)?;
        }
        wtr.flush()?;
    }

    Ok(tmp)
}

fn merge_chunks(
    chunks: Vec<NamedTempFile>,
    headers: &StringRecord,
    output: &PathBuf,
    sort_col: usize,
    dedup: bool,
) -> Result<(), Box<dyn Error>> {
    let mut readers: Vec<_> = chunks
        .iter()
        .map(|f| {
            ReaderBuilder::new()
                .has_headers(true)
                .from_reader(f.reopen().unwrap())
        })
        .collect();

    let mut heap = BinaryHeap::<HeapItem>::new();

    for (idx, rdr) in readers.iter_mut().enumerate() {
        if let Some(result) = rdr.records().next() {
            let rec = result?;
            let key = rec[sort_col].to_string();
            heap.push(HeapItem {
                key,
                chunk_idx: idx,
                record: rec,
            });
        }
    }

    let mut wtr = WriterBuilder::new()
        .has_headers(true)
        .from_path(output)?;
    wtr.write_record(headers)?;

    let mut last_key: Option<String> = None;
    let mut out_count = 0usize;

    while let Some(item) = heap.pop() {
        let emit = if dedup {
            last_key.as_deref() != Some(&item.key)
        } else {
            true
        };

        if emit {
            wtr.write_record(&item.record)?;
            last_key = Some(item.key.clone());
            out_count += 1;

            if out_count % 1_000_000 == 0 {
                println!("已输出 {} 行", out_count);
            }
        }

        let idx = item.chunk_idx;
        if let Some(result) = readers[idx].records().next() {
            let rec = result?;
            let key = rec[sort_col].to_string();
            heap.push(HeapItem {
                key,
                chunk_idx: idx,
                record: rec,
            });
        }
    }

    wtr.flush()?;
    Ok(())
}