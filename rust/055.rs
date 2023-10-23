// 带有生命周期标注的结构体。
#[derive(Debug)]
 struct Borrowed<'a> {
     x: &'a i32,
 }

// 给 impl 标注生命周期。
impl<'a> Default for Borrowed<'a> {
    fn default() -> Self {
        Self {
            x: &10,
        }
    }
}

fn main() {
    let b: Borrowed = Default::default();
    println!("b is {:?}", b);
}
