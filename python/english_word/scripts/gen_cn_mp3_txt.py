import json
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

def main():
    # 读取JSON文件
    with open('/Users/huhao/src/codesnip/python/english_word/scripts/8-1.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 处理每个单元的每个单词
    for unit, words in data.items():
        for word in words:
            if 'chinese' in word:
                processed_text = process_chinese_text(word['chinese'])
                word['cn_mp3_txt'] = processed_text
    
    # 保存修改后的数据回JSON文件
    with open('/Users/huhao/src/codesnip/python/english_word/scripts/8-1.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print("处理完成，已为每个单词添加cn_mp3_txt字段")

if __name__ == "__main__":
    main()