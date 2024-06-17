pub mod lib001;

#[cfg(test)]
mod tests {

    #[test]
    fn it_works() {
        let result = super::lib001::add(2, 2);
        assert_eq!(result, 4);
    }
}
