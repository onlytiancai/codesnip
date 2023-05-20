// 和 C/C++ 不一样，Rust 的函数定义位置是没有限制的
fn main() {
    // 我们可以在这里使用函数，后面再定义它
    fizzbuzz_to(100);
}

// 一个返回布尔值的函数
fn is_divisible_by(lhs: u32, rhs: u32) -> bool {
    // 边界情况，提前返回
    if rhs == 0 {
        return false;
    }

    // 这是一个表达式，可以不用 `return` 关键字
    lhs % rhs == 0
}

// 一个 “不” 返回值的函数。实际上会返回一个单元类型 `()`。
fn fizzbuzz(n: u32) -> () {
    if is_divisible_by(n, 15) {
        println!("fizzbuzz");
    } else if is_divisible_by(n, 3) {
        println!("fizz");
    } else if is_divisible_by(n, 5) {
        println!("buzz");
    } else {
        println!("{}", n);
    }
}

// 当函数返回 `()` 时，函数签名可以省略返回类型
fn fizzbuzz_to(n: u32) {
    for n in 1..=n {
        fizzbuzz(n);
    }
}
