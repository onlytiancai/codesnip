import re

def process_chinese_text(text):
    # 先将中文分号转换为英文分号
    text = text.replace('；', ';')
    
    # 步骤1: 按词性分隔文本
    # 词性模式：以字母开头，包含点、字母、逗号和"&"，然后跟着空格
    # 例如："v. & n. " "prep.,conj. & adv. " "adj. "
    # 使用正则表达式分割，保留分割符
    parts = re.split(r'([a-zA-Z][a-zA-Z.&, ]*\.\s+)', text)
    
    # 合并分割结果，形成词性+中文解释的条目
    # 分割后的结果是：[前导空, 词性1, 中文解释1, 词性2, 中文解释2, ...]
    entries = []
    for i in range(1, len(parts), 2):
        if i+1 < len(parts):
            entries.append(parts[i+1])
    
    # 如果没有匹配到词性，使用原始文本
    if not entries:
        entries = [text]
    
    result = []
    for entry in entries:
        # 对每个中文解释条目进行处理
        # 将连续的多个英文点替换为"什么什么"
        entry = re.sub(r'\.{2,}', '什么什么', entry)
        # 移除英文单个点，但保留中文中点
        entry = entry.replace('.', '')
        # 移除所有非中文、非分号、非"什么什么"、非括号、非问号、非书名号、非中点的字符
        entry = re.sub(r'[^\u4e00-\u9fa5;什么什么()?？·《》（）]', '', entry)
        # 将连续的多个分号替换为单个分号
        entry = re.sub(r';+', ';', entry)
        # 去除首尾分号
        entry = entry.strip(';')
        # 用分号分隔成数组
        if entry:
            items = [item.strip() for item in entry.split(';') if item.strip()]
            result.extend(items)
    
    return result

# 测试用户提供的例子
print("测试1: 名字中间的中点")
test_text1 = "n. 马丁·路德·金"
result1 = process_chinese_text(test_text1)
print("输入:", test_text1)
print("输出:", result1)
print()

print("测试2: 括号内容")
test_text2 = "n. 阿拉巴马州（美国）"
result2 = process_chinese_text(test_text2)
print("输入:", test_text2)
print("输出:", result2)
print()

print("测试3: 问号")
test_text3 = "v. 哪儿不舒服?"
result3 = process_chinese_text(test_text3)
print("输入:", test_text3)
print("输出:", result3)
print()

print("测试4: 书名号")
test_text4 = "n. 《红楼梦》"
result4 = process_chinese_text(test_text4)
print("输入:", test_text4)
print("输出:", result4)
print()

print("测试5: 综合测试")
test_text5 = "n. 马丁·路德·金; 阿拉巴马州（美国）; 哪儿不舒服?; 《红楼梦》"
result5 = process_chinese_text(test_text5)
print("输入:", test_text5)
print("输出:", result5)
print()

print("测试6: 原来的例子")
test_text6 = "conj. 因为; 既然 prep.,conj. & adv. 从......以后; 自......以来"
result6 = process_chinese_text(test_text6)
print("输入:", test_text6)
print("输出:", result6)