#!/usr/bin/env python3
"""
沉浸式 Markdown 翻译工具
使用 OpenAI 兼容 API，保留原有格式，中英文对照翻译
"""

import os
import json
import re
import yaml
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import requests
import time
from dotenv import load_dotenv
import markdown
from markdown.extensions.toc import TocExtension
from markdown.blockprocessors import BlockProcessor
import io

# 加载环境变量
load_dotenv()

@dataclass
class TranslationNode:
    """Markdown 节点"""
    type: str
    content: str
    children: List['TranslationNode'] = None
    level: int = 0
    lang: str = 'en'
    line_number: int = 0

    def __post_init__(self):
        if self.children is None:
            self.children = []

class CustomMarkdownWalker:
    """自定义 Markdown 遍历器，用于流式处理节点"""

    def __init__(self):
        self.nodes = []

    def walk(self, md_text: str) -> List[TranslationNode]:
        """遍历 Markdown 文档"""
        self.nodes = []

        # 先转换成 HTML 了解结构
        md = markdown.Markdown(extensions=['toc', 'fenced_code', 'tables'])
        html = md.convert(md_text)

        # 再逐行解析
        self._parse_lines(md_text)

        return self.nodes

    def _parse_lines(self, md_text: str):
        """逐行解析 Markdown"""
        lines = md_text.split('\n')
        i = 0
        in_code_block = False
        code_block_indent = 0

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # 处理代码块
            if stripped.startswith('```'):
                in_code_block = not in_code_block
                if in_code_block:
                    code_block_indent = len(line) - len(line.lstrip())
                else:
                    # 代码块结束
                    self._add_code_block(lines, i, code_block_indent)
                    i += 1
                i += 1
                continue

            if in_code_block:
                # 继续查找代码块结束
                i += 1
                continue

            # 处理不同类型的行
            if not stripped:  # 空行
                i += 1
                continue

            # 标题
            if re.match(r'^#{1,6}\s', line):
                level = len(re.match(r'^#+', line).group())
                node = TranslationNode('heading', line, level=level, line_number=i)
                self.nodes.append(node)
            # 无序列表
            elif re.match(r'^\s*[-*+]\s', line):
                level = len(line) - len(line.lstrip())
                node = TranslationNode('list', line, level=level, line_number=i)
                self.nodes.append(node)
            # 有序列表
            elif re.match(r'^\s*\d+\.\s', line):
                level = len(line) - len(line.lstrip())
                node = TranslationNode('list', line, level=level, line_number=i)
                self.nodes.append(node)
            # 引用
            elif re.match(r'^\s*>\s', line):
                level = len(line) - len(line.lstrip())
                node = TranslationNode('quote', line, level=level, line_number=i)
                self.nodes.append(node)
            # 代码行
            elif '`' in line and ('`' not in line.strip()[line.strip().find('`')+1:].strip()):
                node = TranslationNode('code_inline', line, line_number=i)
                self.nodes.append(node)
            # 表格
            elif '|' in line and (i == 0 or '|' in lines[i-1] or '|' in lines[i+1] if i+1 < len(lines) else False):
                node = TranslationNode('table', line, line_number=i)
                self.nodes.append(node)
            # 水平线
            elif re.match(r'^\s*[-*_]{3,}\s*$', line):
                node = TranslationNode('hr', line, line_number=i)
                self.nodes.append(node)
            # 普通段落
            else:
                node = TranslationNode('paragraph', line, line_number=i)
                self.nodes.append(node)

            i += 1

    def _add_code_block(self, lines: List[str], start_idx: int, indent: int):
        """添加代码块节点"""
        content = []
        i = start_idx + 1

        while i < len(lines):
            if lines[i].strip().startswith('```'):
                break
            content.append(lines[i])
            i += 1

        # 将代码块内容与起始的 ``` 行合并
        full_content = lines[start_idx] + '\n' + '\n'.join(content)
        node = TranslationNode('code', full_content, line_number=start_idx)
        self.nodes.append(node)

class MarkdownTranslator:
    def __init__(self, api_key: str, api_base: str, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.api_base = api_base.rstrip('/')
        self.model = model
        self.system_prompt = ""
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def analyze_document(self, content: str) -> str:
        """第一步：通读全文，制定翻译思路和风格"""
        prompt = f"""请分析以下 Markdown 文档的内容和风格，制定一个合适的翻译策略。

文档内容：
---
{content}
---

请考虑以下几个方面：
1. 文档类型（技术文档、博客、教程等）
2. 语气和风格（正式、轻松、技术性强等）
3. 专业术语的翻译策略
4. 代码块和特殊内容的处理
5. 保持原有的格式结构

请返回一个系统提示词，用于后续的精确翻译。"""

        try:
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=self.headers,
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 2000
                }
            )
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except Exception as e:
            print(f"分析文档时出错: {e}")
            return "你是一个专业的 Markdown 文档翻译助手。请保持原文的格式、结构和语气，准确翻译内容。代码块和技术术语需要保持原文。"

    def translate_node_streaming(self, node: TranslationNode) -> Tuple[str, str]:
        """第二步：流式翻译单个节点"""
        if node.type == 'code':
            return node.content, node.content

        if not node.content.strip():
            return node.content, node.content

        prompt = f"""{self.system_prompt}

请翻译以下内容，保持原有的 Markdown 格式：

类型：{node.type}
级别：{node.level}
内容：
---
{node.content}
---

请返回两个部分：
1. 原文（保持不变）
2. 中文翻译

格式要求：
- 每个翻译段落后留一个空行
- 保持原文的缩进和格式
- 列表项要一行原文对应一行翻译
- 不翻译代码块"""

        try:
            # 使用流式响应
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=self.headers,
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "stream": True
                }
            )
            response.raise_for_status()

            # 处理流式响应
            translated_content = ""
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line.decode('utf-8'))
                        if 'choices' in chunk and len(chunk['choices']) > 0:
                            delta = chunk['choices'][0].get('delta', {})
                            content = delta.get('content', '')
                            if content:
                                translated_content += content
                                # 实时显示翻译进度
                                print(content, end='', flush=True)
                    except:
                        continue

            print()  # 换行

            return node.content, translated_content.strip()

        except Exception as e:
            print(f"\n翻译时出错: {e}")
            return node.content, node.content

    def parse_markdown(self, content: str) -> List[TranslationNode]:
        """解析 Markdown 为节点树"""
        walker = CustomMarkdownWalker()
        return walker.walk(content)

    def reconstruct_markdown(self, nodes: List[TranslationNode]) -> str:
        """重构翻译后的 Markdown"""
        output = []
        i = 0

        while i < len(nodes):
            node = nodes[i]

            # 添加原文
            if node.type == 'code':
                output.append(node.content)
                i += 1
                continue
            else:
                # 保持原有的缩进
                indent = '  ' * (node.level - 1) if node.level > 1 else ''
                if node.type == 'heading':
                    output.append(node.content)
                elif node.type in ['list', 'quote']:
                    output.append(indent + node.content.strip())
                else:
                    output.append(indent + node.content.strip())

            # 检查是否连续的相同类型节点（用于合并段落）
            j = i + 1
            while j < len(nodes) and nodes[j].type == 'paragraph' and nodes[j].level == node.level:
                # 合并段落
                if nodes[j].type == 'paragraph':
                    output[-1] += '\n' + ('  ' * (nodes[j].level - 1)) + nodes[j].content.strip()
                j += 1

            # 添加空行分隔不同类型的节点
            if j < len(nodes) and nodes[j].type != node.type and nodes[j].type != 'code':
                output.append('')

            # 跳过已合并的节点
            if j > i + 1:
                i = j
            else:
                i += 1

            # 添加翻译
            if node.type != 'code' and hasattr(node, 'translated_content'):
                output.append(node.translated_content)
                output.append('')  # 翻译后加空行

        return '\n'.join(output)

    def translate_markdown(self, input_file: str, output_file: str):
        """翻译 Markdown 文件"""
        # 读取文件
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()

        print("第一步：分析文档风格...")
        self.system_prompt = self.analyze_document(content)
        print("✓ 分析完成，开始翻译...")

        # 解析 Markdown
        nodes = self.parse_markdown(content)

        # 翻译每个节点
        for node in nodes:
            if node.type != 'code':
                print(f"\n正在翻译: {node.type} (级别 {node.level})")
                _, translated = self.translate_node_streaming(node)
                node.translated_content = translated

        # 重新构建 Markdown
        translated_content = self.reconstruct_markdown(nodes)

        # 写入输出文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(translated_content)

        print(f"\n✓ 翻译完成！结果已保存到: {output_file}")

def main():
    # 从环境变量读取配置
    api_key = os.getenv('OPENAI_API_KEY')
    api_base = os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1')
    model = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')

    if not api_key:
        print("错误：请在 .env 文件中设置 OPENAI_API_KEY")
        print("示例：OPENAI_API_KEY=your_api_key")
        return

    # 检查输入文件
    input_file = input("请输入要翻译的 Markdown 文件路径: ").strip()
    if not os.path.exists(input_file):
        print(f"错误：文件不存在 {input_file}")
        return

    # 设置输出文件
    input_path = Path(input_file)
    output_file = input_path.with_suffix(f'{input_path.suffix}_zh-en.md')

    # 创建翻译器
    translator = MarkdownTranslator(api_key, api_base, model)

    # 执行翻译
    translator.translate_markdown(input_file, output_file)

if __name__ == "__main__":
    main()