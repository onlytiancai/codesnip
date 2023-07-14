trait Transpose {
    #[allow(non_snake_case)]
    fn T(&self) -> Vec<Vec<f64>>;
}
type Matrix = Vec<Vec<f64>>;


// 数乘
fn scalar_multiply(matrix: &Matrix, scalar: f64) -> Matrix {
    matrix.iter().map(|row| row.iter().map(|&x| x * scalar).collect()).collect()
}

// 转置
impl Transpose for Matrix { 
    fn T(&self) -> Vec<Vec<f64>> {
        let rows = self.len();
        let cols = self[0].len();
        let mut transposed = vec![vec![0.0; rows]; cols];
        for i in 0..rows {
            for j in 0..cols {
                transposed[j][i] = self[i][j];
            }
        }
        transposed
    }
}


fn inv(matrix: Vec<Vec<f64>>) -> Option<Vec<Vec<f64>>> {
    let n = matrix.len();

    // 构造增广矩阵
    let mut augmented_matrix = vec![vec![0.0; 2 * n]; n];
    for i in 0..n {
        for j in 0..n {
            augmented_matrix[i][j] = matrix[i][j];
        }
        augmented_matrix[i][i + n] = 1.0;
    }

    // 行变换，将增广矩阵变成上三角形矩阵
    for i in 0..n {
        if augmented_matrix[i][i] == 0.0 {
            // 如果主对角线上的元素为零，则矩阵不可逆
            return None;
        }
        for j in i + 1..n {
            let ratio = augmented_matrix[j][i] / augmented_matrix[i][i];
            for k in 0..2 * n {
                augmented_matrix[j][k] -= ratio * augmented_matrix[i][k];
            }
        }
    }

    // 反向代入，将上三角形矩阵变成单位矩阵以及右侧的部分即为逆矩阵
    for i in (0..n).rev() {
        for j in 0..n {
            let mut sum = 0.0;
            for k in i + 1..n {
                sum += augmented_matrix[i][k] * augmented_matrix[k][j + n];
            }
            augmented_matrix[i][j + n] = (augmented_matrix[i][j + n] - sum) / augmented_matrix[i][i];
        }
        augmented_matrix[i][i] = 1.0; // 对角线上的元素变成1
    }

    // 提取逆矩阵部分并返回
    let mut inverse_matrix = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            inverse_matrix[i][j] = augmented_matrix[i][j + n];
        }
    }

    Some(inverse_matrix)
}

// 点乘
fn dot(a: Vec<Vec<f64>>, b: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let rows_a = a.len();
    let cols_a = a[0].len();
    let cols_b = b[0].len();

    let mut result = vec![vec![0.0; cols_b]; rows_a];

    for i in 0..rows_a {
        for j in 0..cols_b {
            for k in 0..cols_a {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    result
}

fn main() {
    let x = vec![
        vec![1.0],
        vec![2.0],
        vec![3.0],
    ];
    let y = scalar_multiply(&x, 2.54);
    println!("X:{:?}", x);
    println!("Y:{:?}", y);
    println!("X.T: {:?}", x.T());
    println!("dot(X.T,X): {:?}", dot(x.T(),x.clone()));
    println!("inv(dot(X.T,X)): {:?}", inv(dot(x.T(),x.clone())));
    /*

    // theta = dot(dot(inv(dot(X.T,X)),X.T),Y)
    match inv(dot(transpose(&x), x.clone())){
        Some(t) =>  {
            let theta = dot(dot(t, transpose(&x)), y);
            println!("result is: {:?}", theta);
        },
        None => println!("inv is none"),
    }
    */
}
