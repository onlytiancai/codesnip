trait Fooable {
    fn foo(&self) -> f64;
}
type Matrix = Vec<Vec<f64>>;

impl Fooable for Matrix{
    fn foo(&self) -> f64{
        0.0
    }
}
fn main() {
    let m = vec![vec![1.0]];
    println!("hello: {:?}", m);
}
