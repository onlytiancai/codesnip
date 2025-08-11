import requests
from bs4 import BeautifulSoup
from readability import Document
from markdownify import markdownify as md

# 要抓取的文章URL
url = 'https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison'

try:
    # 1. 获取网页HTML
    response = requests.get(url, timeout=10)
    response.raise_for_status()

    # --------- 🧹 预处理 HTML ----------
    soup = BeautifulSoup(response.text, 'lxml')

    # 1. 删除常见噪音区域
    for selector in [
        'nav',              # 导航
        'footer',           # 页脚
        'aside',            # 侧边栏
        '.sidebar',         # class 是 sidebar
        '.ads',             # 广告
        '#comments',        # 评论区
        '.newsletter',      # 订阅框
        '[role="navigation"]',
        '[aria-label="breadcrumb"]'
    ]:
        for el in soup.select(selector):
            el.decompose()

    # 2. 可选：移除 script/style 标签
    for tag in soup(["script", "style"]):
        tag.decompose()

    # 3. 转换为字符串
    clean_html = str(soup)

    # 2. 使用readability提取主要内容
    # 将HTML内容传入Document对象
    doc = Document(clean_html)

    # doc.title() 可以获取文章标题
    print(f"文章标题: {doc.title()}\n")

    doc.content()  # 获取干净的HTML内容
    # print("提取的干净HTML内容:\n")
    # print(doc.content())

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