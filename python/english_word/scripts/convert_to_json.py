import re
import json

def convert_txt_to_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    result = {}
    current_unit = None
    
    # 定义正则表达式模式
    unit_pattern = re.compile(r'^Unit\s+(\d+)$')
    # 匹配单词行的模式：单词 [音标] 词性. 中文释义
    word_pattern = re.compile(r'^([a-zA-Z\s\'\-]+?)\s+\[(.+?)\]\s+([a-zA-Z.]+)\.\s+(.+)$')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # 检查是否是单元行
        unit_match = unit_pattern.match(line)
        if unit_match:
            unit_num = unit_match.group(1)
            current_unit = f"Unit {unit_num}"
            result[current_unit] = []
            continue
        
        if current_unit is None:
            continue
        
        # 检查是否包含音标符号
        if '[' in line and ']' in line:
            # 尝试匹配单词行
            word_match = word_pattern.match(line)
            if word_match:
                word = word_match.group(1).strip()
                phonetic = word_match.group(2).strip()
                part_of_speech = word_match.group(3).strip()
                chinese = word_match.group(4).strip()
                
                result[current_unit].append({
                    "word": word,
                    "phonetic": phonetic,
                    "part_of_speech": part_of_speech,
                    "chinese": chinese
                })
            else:
                # 处理包含音标但格式不符合标准的行
                try:
                    # 找到第一个[和最后一个]的位置
                    start_phonetic = line.index('[')
                    end_phonetic = line.rindex(']')
                    
                    # 提取单词/短语部分
                    word_phrase = line[:start_phonetic].strip()
                    # 提取音标部分
                    phonetic = line[start_phonetic+1:end_phonetic].strip()
                    # 提取剩余部分
                    remaining = line[end_phonetic+1:].strip()
                    
                    if '.' in remaining:
                        # 尝试提取词性和中文释义
                        pos_end = remaining.index('.')
                        part_of_speech = remaining[:pos_end].strip()
                        chinese = remaining[pos_end+1:].strip()
                        
                        result[current_unit].append({
                            "word": word_phrase,
                            "phonetic": phonetic,
                            "part_of_speech": part_of_speech,
                            "chinese": chinese
                        })
                    else:
                        # 没有明显的词性标记
                        result[current_unit].append({
                            "phrase": word_phrase,
                            "phonetic": phonetic,
                            "chinese": remaining
                        })
                except:
                    # 如果解析失败，作为特殊行处理
                    result[current_unit].append({
                        "special": line
                    })
        else:
            # 不包含音标，检查是否是特殊行（如人名、书名等）
            if any(keyword in line for keyword in ['/', '（', '）', '(', ')', '《', '》']):
                # 处理包含特殊字符但没有音标的行
                # 尝试识别中文释义的位置（通常中文释义在末尾）
                chinese_start = -1
                for i, char in enumerate(line):
                    # 检查是否是中文字符
                    if '\u4e00' <= char <= '\u9fff':
                        chinese_start = i
                        break
                
                if chinese_start != -1:
                    # 提取短语部分和中文释义部分
                    phrase = line[:chinese_start].strip()
                    chinese = line[chinese_start:].strip()
                    
                    if phrase:
                        result[current_unit].append({
                            "phrase": phrase,
                            "chinese": chinese
                        })
                    else:
                        # 如果短语部分为空，作为特殊行处理
                        result[current_unit].append({
                            "special": line
                        })
                else:
                    # 没有找到中文字符，作为特殊行处理
                    result[current_unit].append({
                        "special": line
                    })
            else:
                # 处理普通短语行
                # 尝试识别中文释义的位置
                chinese_start = -1
                for i, char in enumerate(line):
                    if '\u4e00' <= char <= '\u9fff':
                        chinese_start = i
                        break
                
                if chinese_start != -1:
                    # 提取短语部分和中文释义部分
                    phrase = line[:chinese_start].strip()
                    chinese = line[chinese_start:].strip()
                    
                    if phrase:
                        result[current_unit].append({
                            "phrase": phrase,
                            "chinese": chinese
                        })
                    else:
                        result[current_unit].append({
                            "special": line
                        })
                else:
                    # 没有找到中文字符，可能是纯英文行
                    result[current_unit].append({
                        "word": line,
                        "chinese": ""
                    })
    
    return result

# 转换文件
txt_file = '8-1.txt'
json_data = convert_txt_to_json(txt_file)

# 写入JSON文件
with open('8-1.json', 'w', encoding='utf-8') as f:
    json.dump(json_data, f, ensure_ascii=False, indent=2)

print(f"转换完成！已生成8-1.json文件")