fn main(){
    println!("Hello 2026.");

    let mut arr = [5, 2, 9 ,1, 3];
    println!("before sort");
    print_arr(&arr);

    sort_arr(&mut arr);

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

fn sort_arr<T>(arr: &mut [T]) 
    where T: Ord
{
    for i in 0..arr.len() {
        for j in 0..i {
            if arr[j] > arr[i] {
                arr.swap(i, j)
            }
        }
    }
}