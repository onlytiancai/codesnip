import numpy as np

class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        """
        初始化神经网络的结构和参数。

        参数:
        input_size (int): 输入层神经元的数量。
        hidden_size (int): 隐藏层神经元的数量。
        output_size (int): 输出层神经元的数量。
        learning_rate (float): 学习率。
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # 随机初始化权重和偏置
        # 权重范围通常初始化为较小的随机数，例如 -1 到 1 之间
        # 隐藏层权重 (输入层到隐藏层)
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.b1 = np.zeros((1, self.hidden_size))

        # 输出层权重 (隐藏层到输出层)
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.01
        self.b2 = np.zeros((1, self.output_size))

    def sigmoid(self, x):
        """Sigmoid 激活函数"""
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """Sigmoid 激活函数的导数"""
        return x * (1 - x)

    # 添加 ReLU 激活函数
    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float) # 当 x > 0 时为 1，否则为 0

    def forward(self, X):
        """
        前向传播。

        参数:
        X (np.array): 输入数据，形状为 (样本数, input_size)。

        返回:
        np.array: 神经网络的输出。
        """
        # 隐藏层计算
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1) # 使用 ReLU

        # 输出层计算
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, output):
        """
        反向传播。

        参数:
        X (np.array): 输入数据。
        y (np.array): 真实标签。
        output (np.array): 神经网络的前向传播输出。
        """
        # 计算输出层的误差
        # 使用均方误差 (MSE) 的导数作为输出层的误差项
        # d_loss/d_output = 2 * (output - y) / num_samples
        # 这里我们简化，直接使用 (output - y)
        self.output_error = output - y
        self.output_delta = self.output_error * self.sigmoid_derivative(output)

        # 计算隐藏层的误差
        self.hidden_error = np.dot(self.output_delta, self.W2.T)
        self.hidden_delta = self.hidden_error * self.relu_derivative(self.a1) # 使用 ReLU 导数

        # 更新权重和偏置
        # 权重更新：学习率 * (输入层输出的转置 . 误差项)
        self.W2 -= self.learning_rate * np.dot(self.a1.T, self.output_delta)
        self.b2 -= self.learning_rate * np.sum(self.output_delta, axis=0, keepdims=True)

        self.W1 -= self.learning_rate * np.dot(X.T, self.hidden_delta)
        self.b1 -= self.learning_rate * np.sum(self.hidden_delta, axis=0, keepdims=True)

    def train(self, X, y, epochs):
        """
        训练神经网络。

        参数:
        X (np.array): 训练数据。
        y (np.array): 训练标签。
        epochs (int): 训练迭代次数。
        """
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)

            # 每隔一定epoch打印损失
            if epoch % 1000 == 0:
                loss = np.mean(np.square(y - output)) # 均方误差
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        """
        使用训练好的模型进行预测。

        参数:
        X (np.array): 需要预测的数据。

        返回:
        np.array: 预测结果 (通常经过阈值处理)。
        """
        predictions = self.forward(X)
        # 对于二分类，通常将输出转换为0或1
        return (predictions > 0.5).astype(int)

# --- 示例使用 ---
if __name__ == "__main__":
    # 1. 准备示例数据
    # XOR 逻辑门是一个经典的非线性可分问题，常用于测试简单神经网络
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = np.array([
        [0],
        [1],
        [1],
        [0]
    ])

    # 2. 初始化神经网络
    input_size = 2      # 输入层神经元数量 (XOR 有两个输入)
    hidden_size = 4     # 隐藏层神经元数量 (可以尝试不同值)
    output_size = 1     # 输出层神经元数量 (XOR 有一个输出)
    learning_rate = 0.1 # 学习率

    nn = SimpleNN(input_size, hidden_size, output_size, learning_rate)

    # 3. 训练神经网络
    epochs = 10000 # 训练轮数
    print("开始训练...")
    nn.train(X, y, epochs)
    print("训练完成！")

    # 4. 进行预测
    print("\n预测结果:")
    predictions = nn.predict(X)
    print(f"输入 X:\n{X}")
    print(f"真实 y:\n{y}")
    print(f"预测结果:\n{predictions}")

    # 5. 评估准确率
    accuracy = np.mean(predictions == y)
    print(f"\n准确率: {accuracy * 100:.2f}%")

    # 尝试新的输入
    print("\n对新数据进行预测:")
    new_data = np.array([[0, 1], [1, 1], [0, 0]])
    new_predictions = nn.predict(new_data)
    print(f"新输入:\n{new_data}")
    print(f"新预测:\n{new_predictions}")
