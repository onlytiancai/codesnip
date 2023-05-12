use std::env;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};

fn main() {
    let file_name = env::args().nth(1).unwrap_or_else(|| file!().to_string());
    let f = File::open(file_name).unwrap();
    let reader = BufReader::new(f);

    let mut map = HashMap::new();
    for line in reader.lines() {
        for word in line.unwrap().split_whitespace() {
            *map.entry(String::from(word)).or_insert(0) += 1;
        }
    }

    let mut vec: Vec<(&String, usize)> = map.iter().map(|(k, v)| (k, *v)).collect();
    vec.sort_by(|a, b| b.1.cmp(&a.1));
    for (word, count) in vec.iter().take(5) {
        println!("{}: {}", word, count);
    }
}
