import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号


class LeastSquares:
    def __init__(self):
        self.coefficients = None

    def fit(self, X, y):
        """
        最小二乘法拟合
        X: 自变量数据，形状为 (n_samples, n_features)
        y: 因变量数据，形状为 (n_samples,)
        """
        # 添加偏置项（截距）
        X_with_bias = np.column_stack([np.ones(len(X)), X])

        # 使用最小二乘法公式计算系数
        # θ = (X^T * X)^(-1) * X^T * y
        try:
            XT_X = np.dot(X_with_bias.T, X_with_bias)
            XT_X_inv = np.linalg.inv(XT_X)
            XT_y = np.dot(X_with_bias.T, y)
            self.coefficients = np.dot(XT_X_inv, XT_y)
        except np.linalg.LinAlgError:
            raise ValueError("矩阵不可逆，可能是共线性问题")

        return self

    def predict(self, X):
        """
        预测
        X: 自变量数据
        """
        if self.coefficients is None:
            raise ValueError("模型尚未训练，请先调用fit方法")

        # 添加偏置项
        X_with_bias = np.column_stack([np.ones(len(X)), X])
        return np.dot(X_with_bias, self.coefficients)

    def score(self, X, y):
        """
        计算R²分数
        """
        y_pred = self.predict(X)
        y_mean = np.mean(y)

        ss_total = np.sum((y - y_mean) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)

        if ss_total == 0:
            return 1.0 if ss_residual == 0 else 0.0

        return 1 - (ss_residual / ss_total)


def generate_sample_data(n_samples=100, noise_level=0.1):
    """
    生成示例数据
    """
    np.random.seed(42)
    X = np.random.uniform(0, 10, n_samples)
    true_slope = 2.5
    true_intercept = 1.0
    noise = np.random.normal(0, noise_level, n_samples)
    y = true_slope * X + true_intercept + noise

    return X.reshape(-1, 1), y


def plot_results(X, y, model):
    """
    绘制拟合结果
    """
    plt.figure(figsize=(10, 6))

    # 绘制原始数据点
    plt.scatter(X, y, color='blue', alpha=0.6, label='原始数据')

    # 绘制拟合直线
    X_line = np.array([[0], [10]])
    y_line = model.predict(X_line)
    plt.plot(X_line, y_line, color='red', linewidth=2, label='拟合直线')

    # 添加标签和图例
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('最小二乘法拟合结果')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 显示系数
    intercept, slope = model.coefficients
    plt.text(0.02, 0.98, f'y = {slope:.3f}x + {intercept:.3f}',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.show()


def main():
    # 生成示例数据
    X, y = generate_sample_data(n_samples=100, noise_level=0.5)

    # 创建并训练模型
    model = LeastSquares()
    model.fit(X, y)

    # 打印结果
    print("最小二乘法拟合结果:")
    print(f"截距 (b): {model.coefficients[0]:.4f}")
    print(f"斜率 (w): {model.coefficients[1]:.4f}")

    # 计算R²分数
    r2_score = model.score(X, y)
    print(f"R²分数: {r2_score:.4f}")

    # 绘制结果
    plot_results(X.flatten(), y, model)

    # 梯度下降法实现（对比）
    print("\n梯度下降法实现:")
    gradient_descent(X, y, learning_rate=0.01, n_iterations=1000)


def gradient_descent(X, y, learning_rate=0.01, n_iterations=1000):
    """
    梯度下降法实现最小二乘法
    """
    X_with_bias = np.column_stack([np.ones(len(X)), X.flatten()])
    n_samples = len(X)

    # 初始化参数
    theta = np.random.randn(2)

    for i in range(n_iterations):
        # 计算预测值
        y_pred = np.dot(X_with_bias, theta)

        # 计算误差
        error = y_pred - y

        # 计算梯度
        gradient = (1/n_samples) * np.dot(X_with_bias.T, error)

        # 更新参数
        theta -= learning_rate * gradient

        # 打印进度
        if i % 100 == 0:
            cost = np.sum(error ** 2) / (2 * n_samples)
            print(f"迭代 {i}: cost = {cost:.6f}, theta = [{theta[0]:.4f}, {theta[1]:.4f}]")

    print(f"\n梯度下降法最终结果:")
    print(f"截距 (b): {theta[0]:.4f}")
    print(f"斜率 (w): {theta[1]:.4f}")


if __name__ == "__main__":
    main()