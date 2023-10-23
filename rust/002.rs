#[derive(Debug)]
#[allow(dead_code)]
struct Person<'a> {
    name: &'a str,
    age: u8
}

fn main() {
    let name = "Peter";
    let age = 27;
    #[warn(dead_code)]
    let peter = Person { name, age };

    // Pretty print
    println!("{:#?}", peter);
}
