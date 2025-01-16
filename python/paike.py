import random
import numpy as np

# Constants
NUM_CLASSES = 5  # 班级数量
NUM_PERIODS = 5 * 6  # 一周的时间段数 (5 天，每天 6 节课)
POPULATION_SIZE = 20  # 种群大小
MAX_GENERATIONS = 100  # 最大迭代代数
MUTATION_RATE = 0.1  # 变异概率
CROSSOVER_RATE = 0.7  # 交叉概率

# 示例教师和课程数据
TEACHERS = ["张老师", "李老师", "王老师"]
COURSES = ["语文", "数学", "英语", "音乐", "体育"]
TEACHER_AVAILABILITY = {
    "张老师": {"hard": [("周五", "上午")], "soft": [("周三", "全天")]},
    "李老师": {"hard": [("周三", "下午")], "soft": [("周一", "上午")]},
    "王老师": {"hard": [], "soft": []},
}
COURSE_PREFERENCES = {
    "音乐": {"soft": [("下午", "第一节")], "hard": [("上午", "第一节")]},
}

# 编码
def initialize_population():
    population = []
    for _ in range(POPULATION_SIZE):
        schedule = [
            [
                (random.choice(COURSES), random.choice(TEACHERS))
                for _ in range(NUM_PERIODS)
            ]
            for _ in range(NUM_CLASSES)
        ]
        population.append(schedule)
    return population

# 适应度计算
def calculate_fitness(schedule):
    fitness = 0
    for class_schedule in schedule:
        for period, (course, teacher) in enumerate(class_schedule):
            day = period // 6
            time = period % 6

            # 检查硬约束
            if teacher in TEACHER_AVAILABILITY:
                for hard in TEACHER_AVAILABILITY[teacher]["hard"]:
                    if (day, time) in hard:
                        fitness -= 100

            if course in COURSE_PREFERENCES:
                for hard in COURSE_PREFERENCES[course]["hard"]:
                    if (day, time) in hard:
                        fitness -= 100

            # 软约束
            if teacher in TEACHER_AVAILABILITY:
                for soft in TEACHER_AVAILABILITY[teacher]["soft"]:
                    if (day, time) in soft:
                        fitness += 10

            if course in COURSE_PREFERENCES:
                for soft in COURSE_PREFERENCES[course]["soft"]:
                    if (day, time) in soft:
                        fitness += 10

    return fitness

def select_parents(population, fitness):
    # 确保适应度值非负并归一化
    min_fitness = min(fitness)
    normalized_fitness = [f - min_fitness for f in fitness]
    
    # 检查适应度和是否为零
    if sum(normalized_fitness) == 0:
        # 随机选择父代
        return random.choices(population, k=2)
    else:
        # 按概率选择父代
        probabilities = [f / sum(normalized_fitness) for f in normalized_fitness]
        return random.choices(population, weights=probabilities, k=2)


# 交叉
def crossover(parent1, parent2):
    child1, child2 = parent1[:], parent2[:]
    for i in range(NUM_CLASSES):
        if random.random() < CROSSOVER_RATE:
            child1[i], child2[i] = parent2[i], parent1[i]
    return child1, child2

# 变异
def mutate(schedule):
    for class_schedule in schedule:
        if random.random() < MUTATION_RATE:
            i, j = random.sample(range(NUM_PERIODS), 2)
            class_schedule[i], class_schedule[j] = class_schedule[j], class_schedule[i]
    return schedule

# 遗传算法主循环
def genetic_algorithm():
    population = initialize_population()
    for generation in range(MAX_GENERATIONS):
        fitness = [calculate_fitness(schedule) for schedule in population]
        new_population = []
        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = select_parents(population, fitness)
            child1, child2 = crossover(parent1, parent2)
            new_population.extend([mutate(child1), mutate(child2)])
        population = new_population[:POPULATION_SIZE]

        # 打印当前最佳适应度
        best_fitness = max(fitness)
        print(f"Generation {generation}: Best Fitness = {best_fitness}")

        # 如果满足终止条件，提前结束
        if best_fitness >= 0:
            break

    # 返回最终最佳课表
    best_schedule = population[np.argmax(fitness)]
    return best_schedule

# 运行遗传算法
best_schedule = genetic_algorithm()
print("最佳课表:")
print(best_schedule)

