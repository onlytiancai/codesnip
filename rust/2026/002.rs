
fn main(){
    println!("Hello 2026.");

    let mut arr = [5, 2, 9 ,1, 3];
    println!("before sort");
    let len = arr.len();
    for i in 0..len {
        println!("{}", arr[i]);
    }

    for i in 0..len {
        for j in 0..i {
            if arr[j] > arr[i] {
                let t = arr[i];
                arr[i] = arr[j];
                arr[j] = t;
            }
        }
    }
    println!("after sort");
    for i in 0..len {
        println!("{}", arr[i]);
    }
}