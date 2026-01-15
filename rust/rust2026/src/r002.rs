use std::collections::HashSet;

pub fn main() {
    let str = "When we released Claude Code, we expected developers to use it for coding. They did—and then quickly began using it for almost everything else. This prompted us to build Cowork: a simpler way for anyone—not just developers—to work with Claude in the very same way. Cowork is available today as a research preview for Claude Max subscribers on our macOS app, and we will improve it rapidly from here.";

    let mut  words: HashSet<String> = HashSet::new();
    for word in str
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric())) 
    {

        if !word.is_empty() {
            words.insert(word.to_string());
        }
    }
    println!("word count: {}", words.len());
}