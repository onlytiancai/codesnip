pub fn add(left: usize, right: usize) -> usize {
    left + right
}

pub fn parse(s: &str) -> Vec<String> {
    let mut ret = Vec::new();
    let mut temp = String::new();
    let mut in_number = false;

    for ch in s.chars() {
        if ch.is_ascii_digit() {
            temp.push(ch);
            if !in_number {
                in_number = true;
            }
        } else {
            if in_number {
                in_number = false;
                if !temp.is_empty() {
                    ret.push(temp.clone());
                }
                temp.clear();
            }
            if ch != ' ' {
                ret.push(ch.to_string());
            }
        }
    }

    if !temp.is_empty() {
        ret.push(temp.clone());
    }

    ret
}

#[cfg(test)]
mod tests {

    pub use super::parse;

    #[test]
    fn test_parse() {
        assert_eq!(parse("1+2*3"), vec!["1", "+", "2", "*", "3"]);
    }

    #[test]
    fn test_parse2() {
        assert_eq!(parse("1 + 2 * 3"), vec!["1", "+", "2", "*", "3"]);
    }

    #[test]
    fn test_parse3() {
        assert_eq!(parse("11 + 22 * 33"), vec!["11", "+", "22", "*", "33"]);
    }

    #[test]
    fn test_empty_string() {
        assert_eq!(parse(""), Vec::<String>::new());
    }

    #[test]
    fn test_single_space() {
        assert_eq!(parse(" "), Vec::<String>::new());
    }

    #[test]
    fn test_multiple_spaces() {
        assert_eq!(parse("   "), Vec::<String>::new());
    }

    #[test]
    fn test_single_digit() {
        assert_eq!(parse("5"), vec!["5"]);
    }

    #[test]
    fn test_multiple_digits() {
        assert_eq!(parse("12345"), vec!["12345"]);
    }

    #[test]
    fn test_edge_cases() {
        assert_eq!(parse(" 1 2  34 5   "), vec!["1", "2", "34", "5"]);
    }

}
