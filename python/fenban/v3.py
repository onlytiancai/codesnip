import pygmo as pg
import numpy as np
import random
from collections import defaultdict,namedtuple
import sys
import csv

Student = namedtuple('Student', 'name, gender, total_score, chinese, math, english')
students = []
reader = csv.reader(open('fakedata.csv', encoding='utf-8'))
for i, row in enumerate(reader):
    name = row[0]
    gender = row[1]
    total_score = float(row[2])
    chinese = float(row[3])
    math = float(row[4])
    english = float(row[5])
    print(i, row)
    student = Student(name, gender, total_score, chinese, math, english)
    students.append(student)

print(students)
class_count = 5

def calc(weights):
    # 权重表示每个人所在的班的索引，权重向下取整
    class_list = defaultdict(list)
    for i, x in enumerate(weights):
        class_index = int(x)
        class_list[class_index].append(students[i])

    # 计算出每个班的各科平均成绩，人数，男生人数等
    stu_count_list, boy_count_list = [],[]
    total_score_list, chinese_list, math_list, english_list = [],[],[],[]
    for class_students in class_list.values():
        stu_count_list.append(len(class_students))
        boy_count_list.append(len([stu for stu in class_students if stu.gender=='男']))
        total_score_list.append(round(np.mean([stu.total_score for stu in class_students]),2))
        chinese_list.append(round(np.mean([stu.chinese for stu in class_students]),2))
        math_list.append(round(np.mean([stu.math for stu in class_students]),2))
        english_list.append(round(np.mean([stu.english for stu in class_students]),2))

    return stu_count_list, boy_count_list, total_score_list, chinese_list, math_list, english_list 

# 创建优化问题
class MyProblem:
    def __init__(self, dim):
        self.dim = dim  # 数据维度，学生个数

    def get_nobj(self):
        return 1        # 优化目标个数

    def fitness(self, weights):
        # print(weights)

        stu_count_list, boy_count_list, total_score_list, chinese_list, math_list, english_list = calc(weights)
        '''
        print('*'*20)
        print('各班人数：', stu_count_list, max(stu_count_list)-min(stu_count_list),
            '\n各班男生人数：', boy_count_list, max(boy_count_list)-min(boy_count_list),
              '\n各班总分:', total_score_list, np.std(total_score_list),
              '\n各班语文成绩:', chinese_list,np.std(chinese_list),
              '\n各班数学成绩:', math_list, np.std(math_list),
              '\n各班英语成绩:',english_list, np.std(english_list))
        '''
        return [
                np.std(total_score_list)
                +np.std(chinese_list)
                +np.std(math_list)
                +np.std(english_list)
                + max(stu_count_list) - min(stu_count_list)
                + max(boy_count_list) - min(boy_count_list)
                ]

    def get_bounds(self):
        # 每个维度权重的上下界限，0 到班级个数之间，float 向下取整
        return ([0] * self.dim, [class_count] * self.dim)

# 使用遗传算法进行优化
prob= pg.problem(MyProblem(len(students)))
print(prob)
# algo = pg.algorithm(pg.nsga2(gen=40)) # 多目标算法
algo = pg.algorithm(pg.bee_colony(gen=100, limit = 10))
pop = pg.population(prob, size=40)
status = algo.evolve(pop)

# 获取最优的分班方案
x_list = pop.get_x()
best_x = x_list[pop.best_idx()]
print('best x', [int(x) for x in best_x])

stu_count_list, boy_count_list, total_score_list, chinese_list, math_list, english_list = calc(best_x)
print('*'*20, '最优结果')
print('各班人数：', stu_count_list, max(stu_count_list)-min(stu_count_list),
      '\n各班男生人数：', boy_count_list, max(boy_count_list)-min(boy_count_list),
      '\n各班总分:', total_score_list, np.std(total_score_list),
      '\n各班语文成绩:', chinese_list,np.std(chinese_list),
      '\n各班数学成绩:', math_list, np.std(math_list),
      '\n各班英语成绩:',english_list, np.std(english_list))
best_f = pop.get_f()[pop.best_idx()]
print('best f', best_f)

min_diff = 1000
min_index = -1
for i, x in enumerate(x_list):
    stu_count_list, boy_count_list, total_score_list, chinese_list, math_list, english_list = calc(x)
    diff = max(stu_count_list) - min(stu_count_list)
    if diff < min_diff:
        min_diff = diff
        min_index = i

best_x = x_list[min_index]
stu_count_list, boy_count_list, total_score_list, chinese_list, math_list, english_list = calc(best_x)
print('*'*20, '最优中的最优结果')
print('各班人数：', stu_count_list, max(stu_count_list)-min(stu_count_list),
      '\n各班男生人数：', boy_count_list, max(boy_count_list)-min(boy_count_list),
      '\n各班总分:', total_score_list, np.std(total_score_list),
      '\n各班语文成绩:', chinese_list,np.std(chinese_list),
      '\n各班数学成绩:', math_list, np.std(math_list),
      '\n各班英语成绩:',english_list, np.std(english_list))

