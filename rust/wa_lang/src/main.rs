use std::fmt::Debug;

fn analyze(tokens: Vec<String>) -> Option<Node> {
    let mut token_index = 0;

    fn next(token_index: &mut usize, tokens: &Vec<String>) -> Option<String> {
        if token_index >= &mut tokens.len() {
            None
        } else {
            *token_index += 1;
            Some(tokens[*token_index - 1].clone())
        }
    }

    fn peek(token_index: usize, tokens: &Vec<String>) -> Option<&String> {
        if token_index >= tokens.len() {
            None
        } else {
            Some(&tokens[token_index])
        }
    }


    fn add(token_index: &mut usize, tokens: &Vec<String>) -> Option<Node> {
        let mut left = match mul(token_index, tokens) {
            Some(node) => node,
            None => return None,
        };

        while peek(*token_index, tokens) == Some(&"+".to_string()) {
            next(token_index, tokens);
            left = Node::Binary(Box::new(left), "+".to_string(), Box::new(mul(token_index, tokens)?));
        }

        Some(left)
    }

    fn mul(token_index: &mut usize, tokens: &Vec<String>) -> Option<Node> {
        let mut left = match num(token_index, tokens) {
            Some(node) => node,
            None => return None,
        };

        while peek(*token_index, tokens) == Some(&"*".to_string()) {
            next(token_index, tokens);
            left = Node::Binary(Box::new(left), "*".to_string(), Box::new(mul(token_index, tokens)?));
        }

        Some(left)
    }

    fn num(token_index: &mut usize, tokens: &Vec<String>) -> Option<Node> {
        let token = peek(*token_index, tokens)?;
        if token.chars().all(|c| c.is_digit(10)) {
            next(token_index, tokens);
            Some(Node::Value(token.to_string()))
        } else {
            None
        }
    }

    add(&mut token_index, &tokens)
}

#[derive(Debug)]
enum Node {
    Value(String),
    Binary(Box<Node>, String, Box<Node>),
}

trait DisplayNode: Debug {
    fn display_node(&self, indent: usize) -> String;
}

fn generate_spaces(n: usize) -> String {
    let mut spaces = String::new();
    for _ in 0..n {
        spaces.push('\t');
    }
    spaces
}

impl DisplayNode for Node {
    fn display_node(&self, indent: usize) -> String {
        match self {
            Node::Value(value) => format!("{}{}\n", generate_spaces(indent), value),
            Node::Binary(left, op, right) => format!(
                "{}{}\n{}\n{}\n",
                generate_spaces(indent), op,
                left.display_node(indent+1),
                right.display_node(indent+1)
            ),
        }
    }
}

fn split_string(s: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    for ch in s.chars() {
        tokens.push(ch.to_string());
    }
    tokens
}

fn main() {
    let input = "2+3*4+5";
    let tokens = split_string(&input);
    println!("{:?}\n{:?}", input, tokens);
    let result = analyze(tokens);

    if let Some(node) = result {
        // Handle the AST node (e.g., print it)
        println!("{}", node.display_node(0));
        
    } else {
        println!("Error parsing expression");
    }
}
