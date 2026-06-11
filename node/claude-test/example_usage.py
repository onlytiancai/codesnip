#!/usr/bin/env python3
"""
示例用法：使用 markdown_translator 翻译 1.md
"""

from markdown_translator import MarkdownTranslator
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

def main():
    # 检查 API 配置
    api_key = os.getenv('OPENAI_API_KEY')
    api_base = os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1')
    model = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')

    if not api_key:
        print("错误：请在 .env 文件中设置 OPENAI_API_KEY")
        return

    # 创建翻译器
    translator = MarkdownTranslator(api_key, api_base, model)

    # 输入和输出文件
    input_file = "1.md"
    output_file = "1_translated.md"

    print(f"开始翻译 {input_file}...")
    print("=" * 50)

    try:
        # 执行翻译
        translator.translate_markdown(input_file, output_file)
        print("=" * 50)
        print(f"翻译完成！结果已保存到: {output_file}")
    except Exception as e:
        print(f"翻译过程中出错: {e}")

if __name__ == "__main__":
    main()