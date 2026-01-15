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

    // Check if we need to search in files or stdin
    if args.files.is_empty() {
        // Read from stdin
        search_stdin(&args, &regex_set);
    } else {
        // Search in files
        for file_path in &args.files {
            search_file(&args, &regex_set, file_path);
        }
    }
}

fn search_stdin(args: &Args, regex_set: &regex::RegexSet) {
    let stdin = io::stdin();
    let reader = stdin.lock();
    let mut match_count = 0;

    for line_result in reader.lines() {
        match line_result {
            Ok(line) => {
                let matches = regex_set.is_match(&line);
                let should_print = if args.invert_match { !matches } else { matches };

                if should_print {
                    if args.count {
                        match_count += 1;
                    } else {
                        let output = if args.only_matching {
                            // Find and show only matching parts
                            let re = Regex::new(&args.pattern).unwrap();
                            let matches: Vec<&str> = re.find_iter(&line).map(|m| m.as_str()).collect();
                            matches.join("\n")
                        } else if args.line_number {
                            format!("{}\t{}", match_count + 1, line)
                        } else {
                            line.clone()
                        };
                        println!("{}", output);
                        match_count += 1;
                    }
                } else if args.line_number && args.invert_match {
                    match_count += 1;
                }
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

fn search_file(args: &Args, regex_set: &regex::RegexSet, file_path: &str) {
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
                let matches = regex_set.is_match(&line);
                let should_print = if args.invert_match { !matches } else { matches };

                if should_print {
                    if args.count {
                        match_count += 1;
                    } else {
                        let prefix = if args.files.len() > 1 {
                            format!("{}:", file_path)
                        } else {
                            String::new()
                        };

                        let output = if args.only_matching {
                            // Find and show only matching parts
                            let re = Regex::new(&args.pattern).unwrap();
                            let matches: Vec<&str> = re.find_iter(&line).map(|m| m.as_str()).collect();
                            matches.join("\n")
                        } else if args.line_number {
                            format!("{}{}:{}", prefix, line_count, line)
                        } else {
                            format!("{}{}", prefix, line)
                        };
                        println!("{}", output);
                        match_count += 1;
                    }
                }
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
