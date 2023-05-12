use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};

fn main() {
    let f = File::open(file!()).unwrap();
    let reader = BufReader::new(f);

    let mut map = HashMap::new();
    for line in reader.lines() {
        for word in line.unwrap().split_whitespace() {
            let opt_word: Option<&str> = Option::from(word);
            let word_str = String::from(opt_word.unwrap_or_default());
            *map.entry(word_str).or_insert(0) += 1;
        }
    }

    let mut vec: Vec<(&String, usize)> = map.iter().map(|(k, v)| (k, *v)).collect();
    vec.sort_by(|a, b| b.1.cmp(&a.1));
    for (word, count) in vec.iter().take(5) {
        println!("{}: {}", word, count);
    }
}
