fn main(){
    println!("Hello 2026.");

    let mut arr = [5, 2, 9 ,1, 3];
    println!("before sort");
    print_arr(&arr);

    sort_arr(&mut arr, |a, b| a > b);

    println!("after sort");
    print_arr(&arr);
}

fn print_arr<T>(arr: &[T]) 
    where T: std::fmt::Debug,
{    
    for i in 0..arr.len() {
        println!("{:?}", arr[i]);
    }
}

fn sort_arr<T,F>(arr: &mut [T], mut cmp: F) 
    where F: FnMut(T, T) -> bool,
        T: Clone
{
    for i in 0..arr.len() {
        for j in 0..i {
            if cmp(arr[j].clone(), arr[i].clone()) {
                let t = arr[i].clone();
                arr[i] = arr[j].clone();
                arr[j] = t;
            }
        }
    }
}