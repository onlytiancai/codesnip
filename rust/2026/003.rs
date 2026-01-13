fn main(){
    println!("Hello 2026.");

    let mut arr = [5, 2, 9 ,1, 3];
    println!("before sort");
    print_arr(&arr);

    sort_arr(&mut arr);

    println!("after sort");
    print_arr(&arr);
}

fn greater(a: i32, b:i32) -> bool {
    a > b
}

fn print_arr(arr: &[i32]) {    
    for i in 0..arr.len() {
        println!("{}", arr[i]);
    }
}

fn sort_arr(arr: &mut [i32]) {
    for i in 0..arr.len() {
        for j in 0..i {
            if greater(arr[j], arr[i]) {
                let t = arr[i];
                arr[i] = arr[j];
                arr[j] = t;
            }
        }
    }
}