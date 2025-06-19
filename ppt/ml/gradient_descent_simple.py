import numpy as np

# 定义目标函数 y = x² + sinx
def target_function(x):
    return x**2 + np.sin(x)

# 计算函数在某点的导数（梯度）
def calculate_gradient(x, h=1e-6):
    # 用数值差分法近似导数：f'(x) ≈ [f(x+h) - f(x)] / h
    return (target_function(x + h) - target_function(x)) / h

x = 4 # 随机初始化一个点
learning_rate=0.1            # 初始化一个步长，学习率
min_step=1e-5                # 找到极值最小步长
max_iterations=1000          # 最大迭代次数
last_gradient = calculate_gradient(x)
print(f"随机初始点: x = {x:.4f}, y = {target_function(x):.4f}, last_gradient={last_gradient:.4f}")

for iteration in range(max_iterations):        
    gradient = calculate_gradient(x)     # 计算当前点的梯度
            
    if iteration > 0:                    # 动态调整学习率（步长）            
        if gradient * last_gradient < 0: # 检查梯度方向是否变化
            print(f"学习率变化：gradient={gradient:.4f}, last_gradient={last_gradient:.4f}",
                    f"learning_rate={learning_rate}")
            learning_rate *= 0.5
            
    new_x = x - learning_rate * gradient # 梯度下降更新位置           
    if abs(new_x - x) < min_step:        # 检查是否收敛（步长足够小）
        break
        
    x = new_x
    last_gradient = gradient
    print(f"Iteration {iteration}: x = {x:.4f}, y = {target_function(x):.4f}, "
            f"gradient = {gradient:.4f}, learning_rate = {learning_rate:.4f}")
    
print(f"找到极值点: x = {x:.6f}, y = {target_function(x):.6f}, gradient = {gradient:.6f}")