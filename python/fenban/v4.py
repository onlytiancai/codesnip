'''
https://blog.csdn.net/jnxxhzz/article/details/108637551#0_QQ470585226_2
*****************************************************

代码会自动分配所有学生到每个班，并自动生成每个班级的表格

1. 每个班级男女生数量尽可能平均
2. 每个班级每个分段人数尽可能相等
3. 每个班级之间的所有科目平均分尽可能相近
4. 允许对每个人预设班级

* 分班采用随机算法，每次运行会尝试 20 次计算挑选最小值，多次运算代码可能会得到不同的结果

*****************************************************

需要提供的excel表格内的列大致分为三段：


(姓名 性别)         (语文 数学 英语 科学 总分)    (预设班级)
1. 姓名、性别等信息 ；      2. 成绩 ；          3. 预设班级

*****************************************************

1. 姓名、性别等信息: 可以添加 '姓名' '学号' 等信息，这些信息不会影响结果
                   信息的顺序没有影响，但是此段内容最后一列必须是性别
                   '性别' 的值只能是 '男' 或者 '女'

2. 成绩：顺序无关，但是必须以总分结尾

3. 预设班级: 可选是否存在，需要预设分分班的人后面用数字标明需要分班到哪个班级即可
            注意此列需要使用阿拉伯数字即 1,2,3,4...
            且预设班级的数值应该是 [1,分班数量] 区间内的数字
            不可以超过分班数量 

*****************************************************

举例1：姓名 性别 语文 数学 英语 科学 总分 预设班级

举例2：学号 姓名 性别 数学 语文 英语 科学 总分 预设班级

举例3：姓名 学号 性别 语文 数学 科学 英语 总分

*****************************************************

此项目开源仅仅是为了交流学习，请自觉遵守法律以及道德规范，请勿将其用于商业用途！
有任何问题可以联系QQ470585226

                                        ---by jnxxhzz
                                        杭州二中白马湖学校
'''
import csv

def get_all_students():
    ret = []
    reader = csv.DictReader(open('fakedata_v2.csv', encoding='utf-8'))
    for x in reader:
        x['总分'] = float(x['总分'])
        x['语文'] = float(x['语文'])
        x['数学'] = float(x['数学'])
        x['英语'] = float(x['英语'])
        ret.append(x)
    return ret 

all_students = get_all_students() 
all_students = sorted(all_students, key=lambda x: x['总分'], reverse = True)
need_class = 5 
# print(all_students)

# 初始化每个班级
finall_class = []
every_class = []
for i in range(need_class):
    temp_map = {'男':0,'女':0}
    temp_list = []
    finall_class.append(temp_list)
    every_class.append(temp_map)

# 计算分段人数，每个分段 20 个人
every_level = (int)(20 / need_class) * need_class
if every_level < 20:
    every_level = every_level + need_class

# 蛇形分班 & 标记分段
now_class_number = 0    # 当前学生的班级编号
flag = 1                # 蛇形分布的步长，1 或 -1
level_numebr = 1        # 分数级别，根据条件会递增
now_level_number = 0    # 当前学生的分数级别, 递增
now_every_level = every_level # 每次增加班级个数，或增加 20

boys_number = 0   # 记录男生数量
girls_number = 0  # 记录女生数量

every_level_two = 0

for i in range(len(all_students)):
    # 记录男女生数量
    if all_students[i]['性别'] == '男':
        boys_number = boys_number + 1
        every_class[now_class_number]['男'] = every_class[now_class_number]['男'] + 1
    else:
        girls_number = girls_number + 1
        every_class[now_class_number]['女'] = every_class[now_class_number]['女'] + 1

    # 标记分段
    all_students[i]['分段'] = level_numebr
    now_level_number = now_level_number + 1
    if now_level_number >= now_every_level: 
        if i + 1< len(all_students) and all_students[i + 1]['总分'] == all_students[i]['总分']:
            now_every_level += need_class
        else:
            now_level_number = 0
            level_numebr = level_numebr + 1
            every_level_two = every_level_two + 1
            if every_level_two >= 2:
                now_every_level = now_every_level + every_level
                every_level_two = 0

    # 分班
    finall_class[now_class_number].append(all_students[i]);
    now_class_number = now_class_number + flag
    if now_class_number >= need_class or now_class_number < 0:
        now_class_number = now_class_number - flag
        flag = -flag

import numpy as np
def print_class():
    fenduan_map = {}
    for duan in set([x['分段'] for x in all_students]):
        duan_min = min([float(x['总分']) for x in all_students if x['分段']==duan])
        duan_max = max([float(x['总分']) for x in all_students if x['分段']==duan])
        fenduan_map[duan] = [duan_min, duan_max]

    for i, cla in enumerate(finall_class):
        stu_count = len(cla)
        boy_count = len([x for x in cla if x['性别']=='男'])
        girl_count = len([x for x in cla if x['性别']=='女'])
        total_score = round(np.mean([float(stu['总分']) for stu in cla]),2)
        chinese_score = round(np.mean([float(stu['语文']) for stu in cla]),2)
        math_score = round(np.mean([float(stu['数学']) for stu in cla]),2)
        english_score = round(np.mean([float(stu['英语']) for stu in cla]),2)
        print(f'{i+1} 班:')
        print(f'\t总人数:{stu_count}, 男生人数:{boy_count}，女生人数:{girl_count}')
        print(f'\t语文平均分：{chinese_score}, 数学平均分:{math_score}，英语平均分:{english_score}, 总分平均分:{total_score}')
        for duan, (duan_min, duan_max) in fenduan_map.items():
            duan_count = len([x for x in cla if x['分段']==duan])
            print(f'\t分段{duan}({duan_min}-{duan_max}) 人数：{duan_count}')

print_class()
"[{'男': 22, '女': 18}, {'男': 25, '女': 15}, {'男': 17, '女': 23}, {'男': 17, '女': 23}, {'男': 21, '女': 19}]"

every_boys_number1 = int(sum(x['男'] for x in every_class) / len(every_class))
every_girls_number1 = int(sum(x['女'] for x in every_class) / len(every_class))
every_boys_number2 = every_boys_number1+1
every_girls_number2 = every_girls_number1+1

print('每个班理想男生人数', every_boys_number1, every_boys_number2)
print('每个班理想女生人数', every_girls_number1, every_girls_number2)
print('调整男女比例...')

def has_yushe():
    return False

def change_sex():
    # print(every_boys_number1," ", every_girls_number1)
    # print(every_boys_number2," ", every_girls_number2)

    # 对男女超过平均值的班级配平男女

    # 1. 男女都超过高平均值的两个班级互换
    for boys_id in range(need_class):
        while every_class[boys_id]['男'] > every_boys_number2:
            once_flag = 0
            for girls_id in range(need_class):
                if boys_id != girls_id and every_class[girls_id]['女'] > every_girls_number2:
                    # 在 boys_id 班和 girls_id 班中寻找 分段 相同的男女生交换
                    for boy in range(len(finall_class[boys_id])):
                        if finall_class[boys_id][boy]['性别'] == '男' \
                        and ((has_yushe() and finall_class[boys_id][boy]['预设班级'] == '') or not has_yushe()):
                            for girl in range(len(finall_class[girls_id])):
                                if finall_class[girls_id][girl]['性别'] == '女' \
                                and ((has_yushe() and finall_class[girls_id][girl]['预设班级'] == '') or not has_yushe()) \
                                and finall_class[boys_id][boy]['分段'] ==  finall_class[girls_id][girl]['分段']:
                                    finall_class[boys_id][boy], finall_class[girls_id][girl] = finall_class[girls_id][girl],finall_class[boys_id][boy]
                                    every_class[boys_id]['男'] = every_class[boys_id]['男'] - 1
                                    every_class[boys_id]['女'] = every_class[boys_id]['女'] + 1
                                    every_class[girls_id]['男'] = every_class[girls_id]['男'] + 1
                                    every_class[girls_id]['女'] = every_class[girls_id]['女'] - 1
                                    once_flag = 1;
                                    break
                        if once_flag == 1:
                            break
                if once_flag == 1:
                    break
            if once_flag == 0:
                break
    # 2. 女生超过高平均值的班级和男生超过低平均值的班级互换 （男生班级人数会变成高平均值）
    for girls_id in range(need_class):
        while every_class[girls_id]['女'] > every_girls_number2:
            once_flag = 0
            for boys_id in range(need_class):
                if boys_id != girls_id and every_class[boys_id]['男'] > every_boys_number1:
                    # 在 boys_id 班和 girls_id 班中寻找 分段 相同的男女生交换
                    for boy in range(len(finall_class[boys_id])):
                        if finall_class[boys_id][boy]['性别'] == '男'\
                        and ((has_yushe() and finall_class[boys_id][boy]['预设班级'] == '') or not has_yushe()):
                            for girl in range(len(finall_class[girls_id])):
                                if finall_class[girls_id][girl]['性别'] == '女' \
                                    and ((has_yushe() and finall_class[girls_id][girl]['预设班级'] == '') or not has_yushe()) \
                                    and finall_class[boys_id][boy]['分段'] ==  finall_class[girls_id][girl]['分段']:
                                    finall_class[boys_id][boy], finall_class[girls_id][girl] = finall_class[girls_id][girl],finall_class[boys_id][boy]
                                    every_class[boys_id]['男'] = every_class[boys_id]['男'] - 1
                                    every_class[boys_id]['女'] = every_class[boys_id]['女'] + 1
                                    every_class[girls_id]['男'] = every_class[girls_id]['男'] + 1
                                    every_class[girls_id]['女'] = every_class[girls_id]['女'] - 1
                                    once_flag = 1;
                                    break
                        if once_flag == 1:
                            break
                if once_flag == 1:
                    break
            if once_flag == 0:
                break
    # 3. 男生超过高平均值的班级和女超过低平均值的班级互换 （女班级人数会变成高平均值）
    for boys_id in range(need_class):
        while every_class[boys_id]['男'] > every_boys_number2:
            once_flag = 0
            for girls_id in range(need_class):
                if boys_id != girls_id and every_class[girls_id]['女'] > every_girls_number1:
                    # 在 boys_id 班和 girls_id 班中寻找 分段 相同的男女生交换
                    for boy in range(len(finall_class[boys_id])):
                        if finall_class[boys_id][boy]['性别'] == '男'\
                        and ((has_yushe() and finall_class[boys_id][boy]['预设班级'] == '') or not has_yushe()):
                            for girl in range(len(finall_class[girls_id])):
                                if finall_class[girls_id][girl]['性别'] == '女' \
                                    and ((has_yushe() and finall_class[girls_id][girl]['预设班级'] == '') or not has_yushe()) \
                                    and finall_class[boys_id][boy]['分段'] ==  finall_class[girls_id][girl]['分段']:
                                    finall_class[boys_id][boy], finall_class[girls_id][girl] = finall_class[girls_id][girl],finall_class[boys_id][boy]
                                    every_class[boys_id]['男'] = every_class[boys_id]['男'] - 1
                                    every_class[boys_id]['女'] = every_class[boys_id]['女'] + 1
                                    every_class[girls_id]['男'] = every_class[girls_id]['男'] + 1
                                    every_class[girls_id]['女'] = every_class[girls_id]['女'] - 1
                                    once_flag = 1;
                                    break
                        if once_flag == 1:
                            break
                if once_flag == 1:
                    break
            if once_flag == 0:
                break      

    # 4. 女生低于低平均值的班级和女生等于高平均值班级互换 （男生班级人数会变成低平均值）
    for girls_id in range(need_class):
        while every_class[girls_id]['女'] < every_girls_number1 and every_class[girls_id]['男'] > every_boys_number1:
            once_flag = 0
            for boys_id in range(need_class):
                if boys_id != girls_id and every_class[boys_id]['男'] < every_boys_number2 and every_class[boys_id]['女'] > every_girls_number1:
                    # 在 boys_id 班和 girls_id 班中寻找 分段 相同的男女生交换
                    for boy in range(len(finall_class[girls_id])):
                        if finall_class[girls_id][boy]['性别'] == '男'\
                            and ((has_yushe() and finall_class[girls_id][boy]['预设班级'] == '') or not has_yushe()):
                            for girl in range(len(finall_class[boys_id])):
                                if finall_class[boys_id][girl]['性别'] == '女' \
                                    and ((has_yushe() and finall_class[boys_id][girl]['预设班级'] == '') or not has_yushe()) \
                                    and finall_class[boys_id][girl]['分段'] ==  finall_class[girls_id][boy]['分段']:

                                    finall_class[boys_id][girl], finall_class[girls_id][boy] = finall_class[girls_id][boy],finall_class[boys_id][girl]
                                    every_class[boys_id]['男'] = every_class[boys_id]['男'] + 1
                                    every_class[boys_id]['女'] = every_class[boys_id]['女'] - 1
                                    every_class[girls_id]['男'] = every_class[girls_id]['男'] - 1
                                    every_class[girls_id]['女'] = every_class[girls_id]['女'] + 1
                                    once_flag = 1;
                                    break
                        if once_flag == 1:
                            break
                if once_flag == 1:
                    break
            if once_flag == 0:
                break



change_sex()
print_class()

# 调整预设班级
if has_yushe():
    for i in range(need_class):
        for p1 in range(len(finall_class[i])):
            go_class = finall_class[i][p1]['预设班级']
            if go_class != '' and i != int(go_class) - 1:
                go_class = int(go_class) - 1
                for p2 in range(len(finall_class[go_class])):
                    if finall_class[i][p1]['性别'] == finall_class[go_class][p2]['性别']:
                        finall_class[i][p1], finall_class[go_class][p2] = finall_class[go_class][p2], finall_class[i][p1]
                        break

# 按两个班级的每门课平均分差值是否变小决定是否交换
def check1(max_class_id, p1, min_class_id, p2):
    all_range1 = 0
    for subject in score_key:
        all_range1 = all_range1 + abs(every_class[max_class_id][subject] - every_class[min_class_id][subject])

    all_range2 = 0
    for subject in score_key:
        all_subject = '总分' + subject

        temp1_ave = every_class[max_class_id][all_subject] - finall_class[max_class_id][p1][subject] + finall_class[min_class_id][p2][subject]
        temp1_ave = temp1_ave / len(finall_class[max_class_id])

        temp2_ave = every_class[min_class_id][all_subject] - finall_class[min_class_id][p2][subject] + finall_class[max_class_id][p1][subject]
        temp2_ave = temp2_ave / len(finall_class[min_class_id])

        all_range2 = all_range2 + abs(temp2_ave - temp1_ave)

        # print(all_range1, all_range2)

    if all_range2 < all_range1:
        return True
    else:
        return False

# 按总极差变小决定是否交换
def check2(max_class_id, p1, min_class_id, p2):
    all_range = 0
    all_range1 = 0
    for subject in score_key:
        all_range1 = all_range1 + abs(every_class[max_class_id][subject] - every_class[min_class_id][subject])

    all_range2 = 0
    for subject in score_key:
        all_subject = '总分' + subject

        temp1_ave = every_class[max_class_id][all_subject] - finall_class[max_class_id][p1][subject] + finall_class[min_class_id][p2][subject]
        temp1_ave = temp1_ave / len(finall_class[max_class_id])

        temp2_ave = every_class[min_class_id][all_subject] - finall_class[min_class_id][p2][subject] + finall_class[max_class_id][p1][subject]
        temp2_ave = temp2_ave / len(finall_class[min_class_id])

        if temp1_ave > temp2_ave:
            temp1_ave, temp2_ave = temp2_ave, temp1_ave
        max_score = temp2_ave
        min_score = temp1_ave
        for i in range(need_class):
            if i != max_class_id and i != min_class_id:
                if max_score < every_class[i][subject]:
                    max_score = every_class[i][subject]
                if min_score > every_class[i][subject]:
                    min_score = every_class[i][subject]
        all_range = all_range + max_score - min_score
    # print(all_range1, all_range2)
    return all_range


def change_people(max_class_id, min_class_id, subject):
    global finall_all_range
    for p1 in range(len(finall_class[max_class_id])):
        # 在高分班级中选出高于该科目平均分的人 finall_class[max_class_id][p1]
        if finall_class[max_class_id][p1][subject] > every_class[max_class_id][subject]:
            for p2 in range(len(finall_class[min_class_id])):
                # 预设班级的人不允许交换
                if (book_key.count('预设班级') != 0 and finall_class[max_class_id][p1]['预设班级'] == '' and finall_class[min_class_id][p2]['预设班级'] == '') \
                   or book_key.count('预设班级') == 0:
                    # 在低分班级中选出低于该科目平均分的人 finall_class[min_class_id][p2]
                    if finall_class[max_class_id][p1]['性别'] == finall_class[min_class_id][p2]['性别'] \
                        and finall_class[max_class_id][p1]['分段'] == finall_class[min_class_id][p2]['分段'] \
                        and finall_class[min_class_id][p2][subject] < every_class[min_class_id][subject]:

                        # 计算交换后总极差
                        choice_check = int(random.random() * 2)
                        checkok = False
                        if choice_check == 1:
                            checkok = check1(max_class_id, p1, min_class_id, p2)
                        else:
                            temp_all_range = check2(max_class_id, p1, min_class_id, p2)
                            if temp_all_range < finall_all_range:
                                checkok = True
                        # print(temp_range, finall_all_range)
                        # 若交换后极差变变小则交换
                        if checkok == True:
                            finall_class[max_class_id][p1], finall_class[min_class_id][p2] = finall_class[min_class_id][p2], finall_class[max_class_id][p1]
                            finall_all_range = cal_ave()
