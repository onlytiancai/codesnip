#[derive(Debug)]
struct Centimeters(f64);

#[derive(Debug)]
struct Inches(i32);


impl Inches {
    fn to_centimeters(&self) -> Centimeters {
        let &Inches(inches) = self;
        Centimeters(inches as f64 * 2.54)
    }
}

fn main() {
    let foot = Inches(12);
    let cm = foot.to_centimeters();
    println!("One foot equals {:?} {:?}", foot, cm);
}
