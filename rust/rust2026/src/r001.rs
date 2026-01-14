use std::env;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Error, ErrorKind};

pub fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("please input the file path");
        return Err(Error::new(ErrorKind::Other, "oh no!"));
    }
    let path = &args[1];
    println!("path: {}", path);

    let file = match File::open(path) {
        Ok(f) => f,
        Err(e) => {
            eprint!("file open error");
            return Err(e)
        },
    };
    let reader = BufReader::new(file);
    let mut line_count = 0;
    for _line in reader.lines() {
        line_count += 1;
    }

    println!("line count: {}", line_count);

    Ok(())
}