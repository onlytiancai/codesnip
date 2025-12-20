import re
import json

def convert_txt_to_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    result = {}
    current_unit = None
    unit_counters = {}  # 用于跟踪每个单元的编号计数器
    
    # 定义正则表达式模式
    unit_pattern = re.compile(r'^Unit\s+(\d+)$')
    # 匹配单词行的模式：单词 [音标] 词性. 中文释义，移除词性捕获
    word_pattern = re.compile(r'^([a-zA-Z\s\'\-]+?)\s+\[(.+?)\]\s+(.+)$')
    
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
            unit_counters[current_unit] = 1  # 初始化单元计数器为1
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
                chinese = word_match.group(3).strip()
                
                result[current_unit].append({
                    "id": unit_counters[current_unit],
                    "word": word,
                    "phonetic": phonetic,
                    "chinese": chinese
                })
                unit_counters[current_unit] += 1  # 递增计数器
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
                    # 提取剩余部分（直接作为中文）
                    chinese = line[end_phonetic+1:].strip()
                    
                    result[current_unit].append({
                        "id": unit_counters[current_unit],
                        "word": word_phrase,
                        "phonetic": phonetic,
                        "chinese": chinese
                    })
                    unit_counters[current_unit] += 1  # 递增计数器
                except:
                    # 如果解析失败，作为特殊行处理
                    result[current_unit].append({
                        "id": unit_counters[current_unit],
                        "special": line
                    })
                    unit_counters[current_unit] += 1  # 递增计数器
        else:
            # 不包含音标，检查是否是特殊行（如人名、书名等）
            if any(keyword in line for keyword in ['/', '（', '）', '(', ')', '《', '》']):
                # 处理包含特殊字符但没有音标的行
                # 尝试识别中文释义的位置（通常中文释义在末尾）
                chinese_start = -1
                for i, char in enumerate(line):
                    # 检查是否是中文字符或中文标点符号
                    if ('\u4e00' <= char <= '\u9fff') or char in ['《', '》', '（', '）', '：', '；']:
                        chinese_start = i
                        break
                
                if chinese_start != -1:
                    # 提取短语部分和中文释义部分
                    phrase = line[:chinese_start].strip()
                    chinese = line[chinese_start:].strip()
                    
                    if phrase:
                        result[current_unit].append({
                            "id": unit_counters[current_unit],
                            "word": phrase,  # 使用统一的word字段
                            "phonetic": "",  # 没有音标
                            "chinese": chinese
                        })
                        unit_counters[current_unit] += 1  # 递增计数器
                    else:
                        # 如果短语部分为空，作为特殊行处理
                        result[current_unit].append({
                            "id": unit_counters[current_unit],
                            "special": line
                        })
                        unit_counters[current_unit] += 1  # 递增计数器
                else:
                    # 没有找到中文字符，作为特殊行处理
                    result[current_unit].append({
                        "id": unit_counters[current_unit],
                        "special": line
                    })
                    unit_counters[current_unit] += 1  # 递增计数器
            else:
                # 处理普通短语行
                # 尝试识别中文释义的位置
                chinese_start = -1
                for i, char in enumerate(line):
                    # 检查是否是中文字符或中文标点符号
                    if ('\u4e00' <= char <= '\u9fff') or char in ['《', '》', '（', '）', '：', '；']:
                        chinese_start = i
                        break
                
                if chinese_start != -1:
                    # 提取短语部分和中文释义部分
                    phrase = line[:chinese_start].strip()
                    chinese = line[chinese_start:].strip()
                    
                    if phrase:
                        result[current_unit].append({
                            "id": unit_counters[current_unit],
                            "word": phrase,  # 使用统一的word字段
                            "phonetic": "",  # 没有音标
                            "chinese": chinese
                        })
                        unit_counters[current_unit] += 1  # 递增计数器
                    else:
                        result[current_unit].append({
                            "id": unit_counters[current_unit],
                            "special": line
                        })
                        unit_counters[current_unit] += 1  # 递增计数器
                else:
                    # 没有找到中文字符，可能是纯英文行
                    result[current_unit].append({
                        "id": unit_counters[current_unit],
                        "word": line,
                        "chinese": ""
                    })
                    unit_counters[current_unit] += 1  # 递增计数器
    
    return result

# 转换文件
txt_file = '8-1.txt'
json_data = convert_txt_to_json(txt_file)

# 写入JSON文件
with open('8-1.json', 'w', encoding='utf-8') as f:
    json.dump(json_data, f, ensure_ascii=False, indent=2)

print(f"转换完成！已生成8-1.json文件")