import argparse
import sys
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# matplotlib.rcParams['font.family'] = ['SimHei'] # 或者你系统支持的中文 C:\Windows\Fonts
if sys.platform.startswith('win'):
    matplotlib.rcParams['font.family'] = ['SimHei'] # Windows的中文字体
elif sys.platform.startswith('darwin'):
    matplotlib.rcParams['font.family'] = ['Arial Unicode MS'] # Mac的中文字体
matplotlib.rcParams['axes.unicode_minus'] = False 

def show_digits(digits):
    """显示10个随机手写数字"""
    indices = np.random.choice(len(digits.data), 10, replace=False)
    fig, axes = plt.subplots(2, 5, figsize=(8, 4))
    for i, idx in enumerate(indices):
        ax = axes[i//5, i%5]
        ax.imshow(digits.images[idx], cmap='gray')
        ax.set_title(f'Label: {digits.target[idx]}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def show_predictions(X_test, y_test, y_pred, digits):
    """显示4个测试集图片的真实标签和预测标签"""
    indices = np.random.choice(len(X_test), 4, replace=False)
    fig, axes = plt.subplots(1, 4, figsize=(8, 2))
    for i, idx in enumerate(indices):
        ax = axes[i]
        ax.imshow(X_test[idx].reshape(8, 8), cmap='gray')
        ax.set_title(f'真实: {y_test[idx]}\n预测: {y_pred[idx]}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--show', action='store_true', help='显示10个随机手写数字')
parser.add_argument('--predict', action='store_true', help='显示4个测试集预测结果')
args = parser.parse_args()

# 加载数据
digits = load_digits()
X, y = digits.data, digits.target

# 如果指定了--show参数，显示数字并退出
if args.show:
    show_digits(digits)
    exit()

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建神经网络
mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=42)

# 训练模型
mlp.fit(X_train, y_train)

# 预测
y_pred = mlp.predict(X_test)

# 输出准确率
print(f"准确率: {accuracy_score(y_test, y_pred):.3f}")

# 如果指定了--predict参数，显示预测结果
if args.predict:
    show_predictions(X_test, y_test, y_pred, digits)