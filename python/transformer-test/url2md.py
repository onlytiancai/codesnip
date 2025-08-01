import requests
from readability import Document # 从readability库导入Document
from markdownify import markdownify as md

# 要抓取的文章URL
url = 'https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison'

try:
    # 1. 获取网页HTML
    response = requests.get(url, timeout=10)
    response.raise_for_status()

    # 2. 使用readability提取主要内容
    # 将HTML内容传入Document对象
    doc = Document(response.text, tags_to_keep=['h1', 'ul', 'li', 'p'])

    # doc.title() 可以获取文章标题
    print(f"文章标题: {doc.title()}\n")

    # doc.summary() 返回包含主要内容的HTML片段
    main_content_html = doc.summary() 

    # 3. 将提取出的干净HTML转换为Markdown
    markdown_text = md(main_content_html, heading_style="ATX")

    # 4. 打印结果
    print("--- 主要内容 (Markdown) ---\n")
    print(markdown_text)

except requests.exceptions.RequestException as e:
    print(f"抓取网页时发生错误: {e}")
except Exception as e:
    print(f"处理时发生错误: {e}")