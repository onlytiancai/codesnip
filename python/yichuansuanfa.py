import random

# 遗传算法参数
POPULATION_SIZE = 10  # 种群大小
GENE_LENGTH = 5       # 基因长度（二进制表示的位数）
MUTATION_RATE = 0.1   # 变异概率
GENERATIONS = 20      # 迭代次数

# 目标函数：f(x) = x^2
def fitness_function(x):
    return x ** 2

# 初始化种群
def initialize_population():
    return [random.randint(0, 2**GENE_LENGTH - 1) for _ in range(POPULATION_SIZE)]

# 计算适应度
def calculate_fitness(population):
    return [fitness_function(individual) for individual in population]

# 选择（轮盘赌法）
def select_parents(population, fitness):
    total_fitness = sum(fitness)
    probabilities = [f / total_fitness for f in fitness]
    selected = random.choices(population, weights=probabilities, k=2)
    return selected

# 单点交叉
def crossover(parent1, parent2):
    point = random.randint(1, GENE_LENGTH - 1)
    mask = (1 << point) - 1
    offspring1 = (parent1 & mask) | (parent2 & ~mask)
    offspring2 = (parent2 & mask) | (parent1 & ~mask)
    return offspring1, offspring2

# 变异
def mutate(individual):
    for bit in range(GENE_LENGTH):
        if random.random() < MUTATION_RATE:
            individual ^= (1 << bit)
    return individual

# 主函数
def genetic_algorithm():
    # 初始化种群
    population = initialize_population()
    for generation in range(GENERATIONS):
        # 计算适应度
        fitness = calculate_fitness(population)
        
        # 输出当前代最佳个体
        best_individual = population[fitness.index(max(fitness))]
        print(f"Generation {generation}: Best = {best_individual}, Fitness = {max(fitness)}")
        
        # 新种群
        new_population = []
        while len(new_population) < POPULATION_SIZE:
            # 选择父代
            parent1, parent2 = select_parents(population, fitness)
            # 交叉
            offspring1, offspring2 = crossover(parent1, parent2)
            # 变异
            offspring1 = mutate(offspring1)
            offspring2 = mutate(offspring2)
            # 添加到新种群
            new_population.extend([offspring1, offspring2])
        
        # 更新种群
        population = new_population[:POPULATION_SIZE]

# 运行遗传算法
genetic_algorithm()
