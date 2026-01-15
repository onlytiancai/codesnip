use clap::Parser;
use regex::Regex;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

#[derive(Parser, Debug)]
#[command(name = "grep")]
#[command(about = "A simple grep implementation in Rust", long_about = None)]
struct Args {
    /// Pattern to search for (supports regex)
    #[arg()]
    pattern: String,

    /// Files to search in (if not specified, reads from stdin)
    #[arg()]
    files: Vec<String>,

    /// Print line numbers
    #[arg(short, long)]
    line_number: bool,

    /// Case insensitive search
    #[arg(short, long)]
    ignore_case: bool,

    /// Invert match (select non-matching lines)
    #[arg(short = 'v', long)]
    invert_match: bool,

    /// Print only the count of matching lines
    #[arg(short, long)]
    count: bool,

    /// Show only the matching part of the line
    #[arg(short, long)]
    only_matching: bool,
}

pub fn main() {
    let args = Args::parse();

    // Build regex with case insensitivity option if needed
    let regex_options = if args.ignore_case {
        regex::RegexSetBuilder::new(&[&args.pattern])
            .case_insensitive(true)
            .build()
    } else {
        regex::RegexSet::new(&[&args.pattern])
    };

    let regex_set = match regex_options {
        Ok(set) => set,
        Err(e) => {
            eprintln!("Error compiling regex pattern: {}", e);
            std::process::exit(1);
        }
    };

    // Compile regex for only_matching option (reuse to avoid repeated compilation)
    let regex_for_matching = Regex::new(&args.pattern).ok();

    // Check if we need to search in files or stdin
    if args.files.is_empty() {
        // Read from stdin
        search_stdin(&args, &regex_set, regex_for_matching.as_ref());
    } else {
        // Search in files
        for file_path in &args.files {
            search_file(&args, &regex_set, regex_for_matching.as_ref(), file_path);
        }
    }
}

fn should_process_line(matches: bool, invert_match: bool) -> bool {
    if invert_match { !matches } else { matches }
}

fn format_line_output(
    line: &str,
    line_number: Option<usize>,
    file_prefix: Option<&str>,
    only_matching: bool,
    pattern_regex: Option<&Regex>,
) -> String {
    let prefix = match file_prefix {
        Some(fp) => format!("{}:", fp),
        None => String::new(),
    };

    if only_matching {
        if let Some(re) = pattern_regex {
            let matches: Vec<&str> = re.find_iter(line).map(|m| m.as_str()).collect();
            matches.join("\n")
        } else {
            line.to_string()
        }
    } else {
        match line_number {
            Some(num) => format!("{}{}:{}", prefix, num, line),
            None => format!("{}{}", prefix, line),
        }
    }
}

fn process_line(
    args: &Args,
    line: &str,
    regex_set: &regex::RegexSet,
    regex_for_matching: Option<&Regex>,
    match_count: &mut usize,
    line_number: Option<usize>,
    file_prefix: Option<&str>,
) {
    let matches = regex_set.is_match(line);
    let should_print = should_process_line(matches, args.invert_match);

    if should_print {
        if args.count {
            *match_count += 1;
        } else {
            let output = format_line_output(
                line,
                line_number,
                file_prefix,
                args.only_matching,
                regex_for_matching,
            );
            println!("{}", output);
            *match_count += 1;
        }
    } else if line_number.is_some() && args.invert_match {
        *match_count += 1;
    }
}

fn search_stdin(args: &Args, regex_set: &regex::RegexSet, regex_for_matching: Option<&Regex>) {
    let stdin = io::stdin();
    let reader = stdin.lock();
    let mut match_count = 0;

    for line_result in reader.lines() {
        match line_result {
            Ok(line) => {
                let line_number = if args.line_number { Some(match_count + 1) } else { None };
                process_line(
                    args,
                    &line,
                    regex_set,
                    regex_for_matching,
                    &mut match_count,
                    line_number,
                    None,
                );
            }
            Err(e) => {
                eprintln!("Error reading from stdin: {}", e);
            }
        }
    }

    if args.count {
        println!("{}", match_count);
    }
}

fn search_file(
    args: &Args,
    regex_set: &regex::RegexSet,
    regex_for_matching: Option<&Regex>,
    file_path: &str,
) {
    let path = Path::new(file_path);

    if !path.exists() {
        eprintln!("Error: File '{}' does not exist", file_path);
        return;
    }

    if path.is_dir() {
        eprintln!("Error: '{}' is a directory (use -r for recursive search if implemented)", file_path);
        return;
    }

    let file = match File::open(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Error opening file '{}': {}", file_path, e);
            return;
        }
    };

    let reader = BufReader::new(file);
    let mut match_count = 0;
    let mut line_count = 0;

    for line_result in reader.lines() {
        line_count += 1;
        match line_result {
            Ok(line) => {
                let file_prefix = if args.files.len() > 1 {
                    Some(file_path)
                } else {
                    None
                };

                process_line(
                    args,
                    &line,
                    regex_set,
                    regex_for_matching,
                    &mut match_count,
                    if args.line_number { Some(line_count) } else { None },
                    file_prefix,
                );
            }
            Err(e) => {
                eprintln!("Error reading line {} in file '{}': {}", line_count, file_path, e);
            }
        }
    }

    if args.count {
        if args.files.len() > 1 {
            println!("{}:{}", file_path, match_count);
        } else {
            println!("{}", match_count);
        }
    }
}
