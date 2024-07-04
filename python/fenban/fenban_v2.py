import random
import numpy as np
from deap import base, creator, tools, algorithms

# 示例数据
NUM_STUDENTS = 400
NUM_CLASSES = 10

students = [
    {
        'id': i,
        'gender': random.choice(['M', 'F']),
        'district': random.choice(['A', 'B', 'C']),
        'accommodation': random.choice([True, False]),
        'math': random.randint(50, 100),
        'chinese': random.randint(50, 100),
        'english': random.randint(50, 100)
    }
    for i in range(NUM_STUDENTS)
]

# 创建适应度和个体
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

def evaluate(individual):
    classes = [[] for _ in range(NUM_CLASSES)]
    for i, class_index in enumerate(individual):
        classes[class_index].append(students[i])
    
    scores = []
    for c in classes:
        num_students = len(c)
        num_males = sum(1 for s in c if s['gender'] == 'M')
        num_females = num_students - num_males
        num_accommodations = sum(1 for s in c if s['accommodation'])
        total_scores = [s['math'] + s['chinese'] + s['english'] for s in c]
        avg_math = np.mean([s['math'] for s in c]) if c else 0
        avg_chinese = np.mean([s['chinese'] for s in c]) if c else 0
        avg_english = np.mean([s['english'] for s in c]) if c else 0
        scores.append((
            abs(num_males - num_females),
            abs(num_accommodations - (num_students / 2)),
            np.std(total_scores),
            np.std([s['math'] for s in c]),
            np.std([s['chinese'] for s in c]),
            np.std([s['english'] for s in c])
        ))
    
    total_score = np.sum(scores, axis=0)
    return total_score

# 注册工具
toolbox = base.Toolbox()
toolbox.register("indices", random.choices, range(NUM_CLASSES), k=NUM_STUDENTS)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=0, up=NUM_CLASSES-1, indpb=0.1)
toolbox.register("select", tools.selNSGA2)
toolbox.register("evaluate", evaluate)

# 初始化种群
population = toolbox.population(n=300)

# 进化算法参数
NGEN = 100
CXPB = 0.7
MUTPB = 0.2

# 运行进化算法
for gen in range(NGEN):
    offspring = algorithms.varAnd(population, toolbox, cxpb=CXPB, mutpb=MUTPB)
    fits = map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))

# 获取结果
best_ind = tools.selBest(population, k=1)[0]
classes = [[] for _ in range(NUM_CLASSES)]
for i, class_index in enumerate(best_ind):
    classes[class_index].append(students[i])

# 打印分班结果
for i, c in enumerate(classes):
    print(f"Class {i+1}: {len(c)} students")
