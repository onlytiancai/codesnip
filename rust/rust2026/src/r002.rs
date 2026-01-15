use std::collections::HashMap;

pub fn main() {
    let str = "When we released Claude Code, we expected developers to use it for coding. They did—and then quickly began using it for almost everything else. This prompted us to build Cowork: a simpler way for anyone—not just developers—to work with Claude in the very same way. Cowork is available today as a research preview for Claude Max subscribers on our macOS app, and we will improve it rapidly from here.";

    let mut  words: HashMap<String, i32> = HashMap::new();
    for word in str
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric())) 
    {        
        if !word.is_empty() {
            *words.entry(word.to_string()).or_insert(1) +=1;
        }
    }
    println!("word count: {}", words.len());
    
    let  mut  items: Vec<(&String, &i32)> = words.iter().collect();
    items.sort_by(|a, b| a.1.cmp(b.1));
    for (key, value) in items.iter().take(3)  {
        println!("{}: {}", key, value);
    }
}