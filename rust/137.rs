fn main() {
    let nums = vec![1, 2, 3, 4, 5];
    let nums1 = nums.iter().map(|x| x % 2);
    println!("nums1{:?}",nums1 );

    let nums = vec![1, 2, 3, 4, 5];
    let nums = nums.iter().map(|x| x % 2).collect::<Vec<_>>();
    println!("nums{:?}",nums);

    let a: Vec<Vec<i64>> = vec![vec![1,2,3,4],vec![1,2,3,4]];
    let len1 = a.len();
    let mut remainder_vec = vec![];
    for i in 0..len1{
        let it = a[i].iter().map(|x| x%2).collect::<Vec<_>>();
        remainder_vec.push(it);
    }
    println!("remainder_vec{:?}",remainder_vec);

}
