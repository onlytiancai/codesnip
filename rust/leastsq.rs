use std::ops;

#[derive(Debug,Clone)]
struct Matrix(Vec<Vec<f64>>);

impl From<Vec<Vec<f64>>> for Matrix {
    fn from(item: Vec<Vec<f64>>) -> Self {
        Matrix(item)
    }
}

// 数乘
fn scalar_multiply(matrix: &Vec<Vec<f64>>, scalar: f64) -> Vec<Vec<f64>> {
    // matrix.iter().map(|row| row.iter().map(|&x| x * scalar).collect()).collect()
    let mut ret = vec![];
    for row in matrix {
        let new_row = {
            let mut temp = vec![];
            for col in row {
                temp.push(col * scalar)
            }
            temp
        };
        ret.push(new_row);
    }
    ret
}

impl ops::Mul<f64> for &Matrix {
    type Output = Matrix;

    fn mul(self, rhs: f64) -> Matrix{
        Matrix::from(scalar_multiply(&(self.0), rhs))
    }
}

impl ops::Mul<Matrix> for &Matrix {
    type Output = Matrix;

    fn mul(self, rhs: Matrix) -> Matrix{
        Matrix::from(dot(&(self.0), &(rhs.0)))
    }
}


impl ops::Mul<Matrix> for Matrix {
    type Output = Matrix;

    fn mul(self, rhs: Matrix) -> Matrix{
        Matrix::from(dot(&(self.0), &(rhs.0)))
    }
}

impl ops::Mul<&Matrix> for Matrix {
    type Output = Matrix;

    fn mul(self, rhs: &Matrix) -> Matrix{
        Matrix::from(dot(&(self.0), &(rhs.0)))
    }
}

// 转置
impl Matrix { 
    #[allow(non_snake_case)]
    fn T(&self) -> Matrix {
        let rows = self.0.len();
        let cols = self.0[0].len();
        let mut transposed = vec![vec![0.0; rows]; cols];
        for i in 0..rows {
            for j in 0..cols {
                transposed[j][i] = self.0[i][j];
            }
        }
        Matrix::from(transposed)
    }

    fn inv(&self) -> Matrix {
        inv(self).unwrap()
    }
}


fn inv(matrix: &Matrix) -> Option<Matrix> {
    let n = matrix.0.len();

    println!("before augmented_matrix: {:?}", matrix);
    // 构造增广矩阵
    let mut augmented_matrix = vec![vec![0.0; 2 * n]; n];
    for i in 0..n {
        for j in 0..n {
            augmented_matrix[i][j] = matrix.0[i][j];
        }
        augmented_matrix[i][i + n] = 1.0;
    }
    println!("augmented_matrix: {:?}", augmented_matrix);

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
    println!("triangle augmented_matrix: {:?}", augmented_matrix);
    

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

    Some(Matrix(inverse_matrix))
}

// 点乘
fn dot(a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
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

fn leastsq(x: &Matrix, y: &Matrix) -> Matrix {
    (x.T() * x).inv() * x.T() * y
}


fn main() {
    let x = Matrix(vec![
        vec![1.3, 2.3,3.4],
        vec![2.9,3.5,4.7],
        vec![3.6,2.8,9.5],
    ]);
    let y = &x * Matrix(vec![vec![0.8], vec![7.9], vec![6.5]]);
    let theta = (x.T() * &x).inv() * x.T() * &y;
    println!("result is: {:?}", theta);

    let theta = leastsq(&x, &y); 
    println!("result is: {:?}", theta);

    println!("X:{:?}", x);
    println!("X:{:?}", y);
    println!("X.T: {:?}", x.T());
    println!("dot(X.T,X): {:?}", x.T() * &x);
    println!("inv(dot(X.T,X)): {:?}", (x.T() * x).inv());

}
