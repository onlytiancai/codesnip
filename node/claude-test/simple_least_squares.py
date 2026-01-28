import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号


def least_squares_fit(x, y):
    """
    最小二乘法拟合直线
    y = wx + b
    """
    # 计算必要的和
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x ** 2)

    # 计算斜率和截距
    w = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
    b = (sum_y - w * sum_x) / n

    return w, b


def r_squared(y_true, y_pred):
    """
    计算R²分数
    """
    y_mean = np.mean(y_true)
    ss_total = np.sum((y_true - y_mean) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)

    return 1 - (ss_residual / ss_total)


def main():
    # 生成示例数据
    np.random.seed(42)
    x = np.random.uniform(0, 10, 50)
    y = 2.5 * x + 1.0 + np.random.normal(0, 0.5, 50)  # 真实斜率2.5，截距1.0

    # 使用最小二乘法拟合
    w, b = least_squares_fit(x, y)

    print("最小二乘法拟合结果:")
    print(f"y = {w:.4f}x + {b:.4f}")

    # 计算预测值和R²
    y_pred = w * x + b
    r2 = r_squared(y, y_pred)
    print(f"R²分数: {r2:.4f}")

    # 可视化
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', label='原始数据', alpha=0.6)
    plt.plot(x, y_pred, color='red', linewidth=2, label='拟合直线')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('最小二乘法拟合')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 添加方程文本
    plt.text(0.02, 0.98, f'y = {w:.3f}x + {b:.3f}\nR² = {r2:.3f}',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.show()


if __name__ == "__main__":
    main()