import re
import json

# 定义正则表达式模式
UNIT_PATTERN = re.compile(r'^Unit\s+(\d+)$')
# 匹配单词行的模式：单词 [音标] 中文释义
WORD_PATTERN = re.compile(r'^([a-zA-Z\s\'\-]+?)\s+\[(.+?)\]\s+(.+)$')


def _find_chinese_start(line):
    """查找行中中文开始的位置
    
    Args:
        line: 待检查的字符串
        
    Returns:
        中文开始的索引位置，如果没有找到则返回 -1
    """
    for i, char in enumerate(line):
        # 检查是否是中文字符或中文标点符号
        if ('\u4e00' <= char <= '\u9fff') or char in ['《', '》', '（', '）', '：', '；']:
            return i
    return -1


def _parse_unit_line(line, result, unit_counters):
    """解析单元行
    
    Args:
        line: 待解析的行
        result: 当前结果字典
        unit_counters: 单元计数器字典
        
    Returns:
        解析成功的单元名称，否则返回 None
    """
    unit_match = UNIT_PATTERN.match(line)
    if unit_match:
        unit_num = unit_match.group(1)
        current_unit = f"Unit {unit_num}"
        result[current_unit] = []
        unit_counters[current_unit] = 1  # 初始化单元计数器为1
        return current_unit
    return None


def _parse_word_with_phonetic(line):
    """解析带有音标的单词行
    
    Args:
        line: 待解析的行
        
    Returns:
        包含 word、phonetic 和 chinese 的字典，如果解析失败则返回 None
    """
    # 尝试匹配单词行
    word_match = WORD_PATTERN.match(line)
    if word_match:
        return {
            "word": word_match.group(1).strip(),
            "phonetic": word_match.group(2).strip(),
            "chinese": word_match.group(3).strip()
        }
    
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
        
        return {
            "word": word_phrase,
            "phonetic": phonetic,
            "chinese": chinese
        }
    except:
        # 如果解析失败，返回 None
        return None


def _parse_phrase_without_phonetic(line):
    """解析不带有音标的短语行
    
    Args:
        line: 待解析的行
        
    Returns:
        包含 word、phonetic 和 chinese 的字典，如果解析失败则返回 None
    """
    # 尝试识别中文释义的位置
    chinese_start = _find_chinese_start(line)
    
    if chinese_start != -1:
        # 提取短语部分和中文释义部分
        phrase = line[:chinese_start].strip()
        chinese = line[chinese_start:].strip()
        
        if phrase:
            return {
                "word": phrase,
                "phonetic": "",  # 没有音标
                "chinese": chinese
            }
    
    # 没有找到中文字符，可能是纯英文行
    return {
        "word": line,
        "phonetic": "",
        "chinese": ""
    }


def convert_txt_to_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    result = {}
    current_unit = None
    unit_counters = {}  # 用于跟踪每个单元的编号计数器
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # 检查是否是单元行
        unit_result = _parse_unit_line(line, result, unit_counters)
        if unit_result:
            current_unit = unit_result
            continue
        
        if current_unit is None:
            continue
        
        # 检查是否包含音标符号
        if '[' in line and ']' in line:
            word_data = _parse_word_with_phonetic(line)
            if word_data:
                result[current_unit].append({
                    "id": unit_counters[current_unit],
                    **word_data
                })
                unit_counters[current_unit] += 1  # 递增计数器
            else:
                # 如果解析失败，作为特殊行处理
                result[current_unit].append({
                    "id": unit_counters[current_unit],
                    "special": line
                })
                unit_counters[current_unit] += 1  # 递增计数器
        else:
            # 不包含音标，解析短语行
            phrase_data = _parse_phrase_without_phonetic(line)
            
            # 检查是否是包含特殊字符但解析失败的情况
            if not phrase_data["word"] and any(keyword in line for keyword in ['/', '（', '）', '(', ')', '《', '》']):
                # 作为特殊行处理
                result[current_unit].append({
                    "id": unit_counters[current_unit],
                    "special": line
                })
            else:
                result[current_unit].append({
                    "id": unit_counters[current_unit],
                    **phrase_data
                })
            
            unit_counters[current_unit] += 1  # 递增计数器
    
    return result

import unittest


class TestWordParser(unittest.TestCase):
    """单元测试类"""
    
    def test_find_chinese_start(self):
        """测试中文开始位置查找"""
        self.assertEqual(_find_chinese_start("Journey to the West 《西游记》"), 20)
        self.assertEqual(_find_chinese_start("feel free （可以）随便（做某事）"), 10)
        self.assertEqual(_find_chinese_start("hello [həˈləʊ] 你好"), 15)
        self.assertEqual(_find_chinese_start("pure english"), -1)
        
    def test_parse_unit_line(self):
        """测试单元行解析"""
        result = {}
        unit_counters = {}
        
        # 测试正常单元行
        unit = _parse_unit_line("Unit 1", result, unit_counters)
        self.assertEqual(unit, "Unit 1")
        self.assertIn("Unit 1", result)
        self.assertEqual(result["Unit 1"], [])
        self.assertEqual(unit_counters["Unit 1"], 1)
        
        # 测试不是单元行
        unit = _parse_unit_line("not a unit", result, unit_counters)
        self.assertIsNone(unit)
    
    def test_parse_word_with_phonetic(self):
        """测试带音标单词解析"""
        # 测试标准格式
        result = _parse_word_with_phonetic("hello [həˈləʊ] 你好")
        self.assertEqual(result["word"], "hello")
        self.assertEqual(result["phonetic"], "həˈləʊ")
        self.assertEqual(result["chinese"], "你好")
        
        # 测试复杂格式
        result = _parse_word_with_phonetic("break down [ˈbreɪk daʊn] 分解；发生故障")
        self.assertEqual(result["word"], "break down")
        self.assertEqual(result["phonetic"], "ˈbreɪk daʊn")
        self.assertEqual(result["chinese"], "分解；发生故障")
    
    def test_parse_phrase_without_phonetic(self):
        """测试无音标短语解析"""
        # 测试书名号格式
        result = _parse_phrase_without_phonetic("Journey to the West 《西游记》")
        self.assertEqual(result["word"], "Journey to the West")
        self.assertEqual(result["phonetic"], "")
        self.assertEqual(result["chinese"], "《西游记》")
        
        # 测试括号格式
        result = _parse_phrase_without_phonetic("feel free （可以）随便（做某事）")
        self.assertEqual(result["word"], "feel free")
        self.assertEqual(result["phonetic"], "")
        self.assertEqual(result["chinese"], "（可以）随便（做某事）")
        
        # 测试普通中文格式
        result = _parse_phrase_without_phonetic("good morning 早上好")
        self.assertEqual(result["word"], "good morning")
        self.assertEqual(result["phonetic"], "")
        self.assertEqual(result["chinese"], "早上好")
        
        # 测试纯英文
        result = _parse_phrase_without_phonetic("pure english")
        self.assertEqual(result["word"], "pure english")
        self.assertEqual(result["phonetic"], "")
        self.assertEqual(result["chinese"], "")


if __name__ == "__main__":
    import sys
    
    # 检查是否有--test参数
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # 运行单元测试
        unittest.main(argv=[sys.argv[0]], exit=False)
    else:
        # 正常转换文件
        txt_file = '8-1.txt'
        json_data = convert_txt_to_json(txt_file)
        
        # 写入JSON文件
        with open('8-1.json', 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        print(f"转换完成！已生成8-1.json文件")