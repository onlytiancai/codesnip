import random
import math

# 定义参数
POPULATION_SIZE = 100
GENE_LENGTH = 10  # 假设我们使用10位精度
GENERATIONS = 1000
TARGET = 25.0

def initialize_population():
    return [{'chromosome': [random.randint(0, 9) for _ in range(GENE_LENGTH)], 'fitness': 0.0} for _ in range(POPULATION_SIZE)]

def decode(chromosome):
    return sum([chromosome[i] * (10 ** (GENE_LENGTH - 1 - i)) for i in range(GENE_LENGTH)]) / (10 ** GENE_LENGTH)

def fitness(chromosome):
    decoded = decode(chromosome)
    return 1 / (abs(decoded ** 2 - TARGET) + 1e-8)  # 避免除以零，加小数防止极端情况

def selection(population):
    population.sort(key=lambda x: x['fitness'], reverse=True)
    return population[:POPULATION_SIZE // 2]

def crossover(parent1, parent2):
    point = random.randint(1, GENE_LENGTH - 1)
    child_chromosome = parent1['chromosome'][:point] + parent2['chromosome'][point:]
    return {'chromosome': child_chromosome, 'fitness': 0.0}

def mutation(individual, mutation_rate=0.01):
    for i in range(GENE_LENGTH):
        if random.random() < mutation_rate:
            individual['chromosome'][i] = random.randint(0, 9)
    return individual

def genetic_algorithm():
    population = initialize_population()
    
    for generation in range(GENERATIONS):
        for individual in population:
            individual['fitness'] = fitness(individual['chromosome'])
        
        selected = selection(population)
        new_population = selected.copy()
        
        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = random.sample(selected, 2)
            child = crossover(parent1, parent2)
            child = mutation(child)
            new_population.append(child)
        
        population = new_population
        
        # 每100代打印一次最佳结果
        if generation % 100 == 0:
            best = max(population, key=lambda x: x['fitness'])
            print(f"Generation {generation}: Best solution = {decode(best['chromosome']):.6f}, Fitness = {best['fitness']:.6f}")
    
    best = max(population, key=lambda x: x['fitness'])
    return decode(best['chromosome'])

# 运行遗传算法
result = genetic_algorithm()
print(f"Approximation of the square root of 25: {result:.6f}")
print(f"Actual square root of 25: {math.sqrt(25):.6f}")
