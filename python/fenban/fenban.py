import pygmo as pg
import numpy as np
import random

# 定义学生类
class Student:
    def __init__(self, gender, district, is_boarding, math, chinese, english):
        self.gender = gender
        self.district = district
        self.is_boarding = is_boarding
        self.math = math
        self.chinese = chinese
        self.english = english
        self.total_score = math + chinese + english
    def __repr__(self):
        return f'{self.gender} {self.math} {self.chinese} {self.english}'

# 生成学生数据
students = []
for i in range(400):
    gender = random.choice(['男', '女'])
    district = random.choice(['A', 'B', 'C'])
    is_boarding = random.choice([True, False])
    math = random.randint(60, 100)
    chinese = random.randint(60, 100)
    english = random.randint(60, 100)
    student = Student(gender, district, is_boarding, math, chinese, english)
    students.append(student)

print(students)

# 定义适应度函数
def fitness_function(x):
    # x 表示一个分班方案
    # 计算每个班级的人数、男女比例、住宿比例、总分平均值以及每科平均成绩
    # 根据约束条件和目标计算每个方案的适应度
    # 返回一个包含各项指标的适应度向量

    # 计算班级人数
    class_sizes = np.bincount(chromosome)

    # 计算男女比例
    male_ratio = np.sum(students[chromosome == 0, 0]) / class_sizes[0]
    female_ratio = np.sum(students[chromosome == 1, 0]) / class_sizes[1]

    # 计算住宿比例
    boarder_ratio = np.sum(students[chromosome == 2, 2]) / class_sizes[2]
    day_ratio = np.sum(students[chromosome == 3, 2]) / class_sizes[3]

    # 计算总分
    total_scores = np.sum(students[:, 3:], axis=1)
    class_total_scores = np.bincount(chromosome, weights=total_scores)

    # 计算各科平均分
    subject_means = np.zeros(3)
    for i in range(3):
        subject_means[i] = np.bincount(chromosome, weights=students[:, i+3]) / class_sizes

    # 定义目标函数值
    f1 = np.std(class_sizes)
    f2 = np.abs(male_ratio - female_ratio)
    f3 = np.abs(boarder_ratio - day_ratio)
    f4 = np.std(class_total_scores)
    f5 = np.std(subject_means)
    return f1 + f2 + f3 + f4 + f5


# 创建优化问题
class MyProblem:
    def __init__(self):
        self.dim = 400  # 变量的维度
        self.obj_dim = 5  # 目标的维度

    def fitness(self, x):
        return fitness_function(x)

    def get_bounds(self):
        # 定义每个变量的取值范围

# 使用遗传算法进行优化
prob = pg.problem(MyProblem())
algo = pg.algorithm(pg.nsga2(gen=100))
pop = pg.population(prob, size=100)
status = algo.evolve(pop)

# 获取最优的分班方案
best_solution = status.get_f()
print('best solution', best_solution)

best_chromosome = pop.champion.chrom
class_assignments = best_chromosome.x
for i, class_id in enumerate(class_assignments):
    print(f"学生 {i} 分配到第 {class_id + 1} 班")
