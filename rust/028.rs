// 定义一个函数，可以接受一个由 `Fn` 限定的泛型 `F` 参数并调用它。
fn call_me<F: Fn()>(f: F) {
    f()
}

fn apply<F>(f: F) where
    F: Fn() {
    f();
}

// 定义一个满足 `Fn` 约束的封装函数（wrapper function）。
fn function() {
    println!("I'm a function!");
}

fn create_fn() -> impl Fn() {
    let text = "Fn".to_owned();

    move || println!("This is a: {}", text)
}

fn create_fnmut() -> impl FnMut() {
    let mut text = "FnMut".to_owned();

    move || {
        text.push_str(" 111");
        println!("This is a: {}", text);
    }
}

fn create_fnonce() -> impl FnOnce() {
    let text = "FnOnce".to_owned();
    
    move || println!("This is a: {}", text)
}



fn main() {
    // 定义一个满足 `Fn` 约束的闭包。
    let closure = || println!("I'm a closure!");
    
    call_me(closure);
    call_me(function);

    apply(closure);
    apply(function);

    let fn_plain = create_fn();
    let mut fn_mut = create_fnmut();
    let fn_once = create_fnonce();

    fn_plain();
    fn_plain();
    fn_mut();
    fn_mut();
    fn_mut();
    fn_once();
    // fn_once();


}

