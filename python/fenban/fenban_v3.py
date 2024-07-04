import pygmo as pg
import numpy as np
import random
from collections import defaultdict

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

students = []
obj_dim = 10
for i in range(obj_dim):
    gender = random.choice(['男', '女'])
    district = random.choice(['A', 'B', 'C'])
    is_boarding = random.choice([True, False])
    math = random.randint(60, 100)
    chinese = random.randint(60, 100)
    english = random.randint(60, 100)
    student = Student(gender, district, is_boarding, math, chinese, english)
    students.append(student)

print(students)

# 创建优化问题
class MyProblem:
    def __init__(self):
        self.obj_dim = obj_dim # 目标维度

    def get_nobj(self):
        return 2    # 优化目标个数 

    def fitness(self, weights):
        print(weights)
        # 权重表示每个人所在的班的索引
        class_list = defaultdict(list)
        for i, x in enumerate(weights):
            class_index = int(x)
            class_list[class_index].append(students[i])

        # 计算出每个班的数学，英语成绩
        math_list, english_list = [], []
        for class_students in class_list.values():
            math_list.append(sum(stu.math for stu in class_students))
            english_list.append(sum(stu.english for stu in class_students))

        # 让各班之间的数学，英语成绩标准差最小
        return [np.std(math_list), np.std(english_list)]

    def get_bounds(self):
        return ([0] * self.obj_dim, [5] * self.obj_dim)

# 使用遗传算法进行优化
prob = pg.problem(MyProblem())
print(prob)
algo = pg.algorithm(pg.nsga2(gen=100))
pop = pg.population(prob, size=40)
status = algo.evolve(pop)

# 获取最优的分班方案
best_solution = status.get_f()
print('best solution', best_solution)
print(status)
