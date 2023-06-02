#[derive(Debug)]
struct Centimeters<'a>(&'a f64);

#[derive(Debug)]
struct Inches(i32);


impl Inches {
    fn to_centimeters<'a>(&'a self, g: &'a mut f64) -> Centimeters {
        let &Inches(inches) = self;
        *g = inches as f64 * 2.54;
        Centimeters(g)
    }
}

fn main() {
    let foot = Inches(12);
    let mut g:f64 = 0.0;
    let cm = foot.to_centimeters(&mut g);
    println!("One foot equals {:?} {:?}", foot, cm);
}
