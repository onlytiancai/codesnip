// 导入标准库中的 env 模块，用于处理程序的环境变量
use std::env; 
// 导入标准库中的 collections 模块中的 HashMap 类型，用于存储字符串和它们的计数值
use std::collections::HashMap; 
// 导入标准库中的 fs 模块中的 File 类型，用于打开文件流
use std::fs::File; 
// 导入标准库中的 io 模块中的 BufRead 和 BufReader 类型，用于按行读取文件流，BufRead 不能省略
use std::io::{BufRead, BufReader}; 

fn main() {
    // 获取命令行参数中的文件名，如果没有则使用当前文件名
    let file_name = env::args().nth(1).unwrap_or_else(|| file!().to_string()); 
    // 打开文件流，如果出错则 panic
    let f = File::open(file_name).unwrap(); 
    // 创建一个按行读取的缓冲读取器
    let reader = BufReader::new(f); 
    // 创建一个空的 HashMap，用于存储单词计数
    let mut map = HashMap::new(); 

    // 遍历读取器中的每一行
    for line in reader.lines() {
        // 将每一行按空格分割后遍历每一个单词
        for word in line.unwrap().split_whitespace() {
            // 创建一个 String 类型的变量，存储 word 的值
            let word_str = String::from(word);
            // 获取 word 对应的计数值，如果 word 不存在则插入一个值为 0 的键值对，并返回该键的计数值
            let count = map.entry(word_str).or_insert(0);
            // 将计数值加 1
            *count += 1;
        }
    }

    // 将 HashMap 转换成 Vec 中元素为元组类型，第一个元素为字符串，第二个元素为计数值，并按计数值从大到小排序
    let mut vec: Vec<(&String, usize)> = map.iter().map(|(k, v)| (k, *v)).collect();
    vec.sort_by(|a, b| b.1.cmp(&a.1));

    // 遍历排序后的 Vec，输出前五个单词和它们的计数值
    for (word, count) in vec.iter().take(5) {
        println!("{}: {}", word, count);
    }
}
