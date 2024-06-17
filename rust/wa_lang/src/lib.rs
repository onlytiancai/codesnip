pub mod lib001;

#[cfg(test)]
mod tests {

    pub use super::lib001::add;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }

}
