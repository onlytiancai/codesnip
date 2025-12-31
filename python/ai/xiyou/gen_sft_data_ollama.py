import json
import re
import requests
from tqdm import tqdm

# ====== 配置区 ======
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:4b"

TEXT_PATH = "./data/xiyou_raw.txt"
OUTPUT_PATH = "xiyou_instruct.jsonl"

CHUNK_SIZE = 600
STRIDE = 100
# ====================


def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    text = re.sub(r"\n{2,}", "\n", text)
    return text


def split_text(text, chunk_size=600, stride=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if len(chunk) > 200:
            chunks.append(chunk)
        start += chunk_size - stride
    return chunks


def build_prompt(text):
    return f"""你是中文古典文学助教。
请根据下面的《西游记》原文，生成 3 组高质量问答。

要求：
1. 问题必须基于原文内容
2. 回答使用现代汉语
3. 不引入原文以外的知识
4. 输出格式如下：

Q1: ...
A1: ...
Q2: ...
A2: ...
Q3: ...
A3: ...

原文：
{text}
"""


def call_ollama(prompt):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.6,
            "top_p": 0.9
        }
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()["response"]


def parse_qa(output_text):
    qa_pairs = []
    pattern = re.compile(r"Q\d+:(.*?)A\d+:(.*?)(?=Q\d+:|$)", re.S)
    for q, a in pattern.findall(output_text):
        q = q.strip()
        a = a.strip()
        if len(q) > 5 and len(a) > 5:
            qa_pairs.append((q, a))
    return qa_pairs


def main():
    text = load_text(TEXT_PATH)
    chunks = split_text(text)

    print(f"共切分得到 {len(chunks)} 段文本")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
        for idx, chunk in enumerate(tqdm(chunks)):
            prompt = build_prompt(chunk)

            try:
                output = call_ollama(prompt)
            except Exception as e:
                print(f"第 {idx} 段失败：{e}")
                continue

            qa_pairs = parse_qa(output)

            for q, a in qa_pairs:
                record = {
                    "instruction": q,
                    "input": "",
                    "output": a
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()

    print("数据生成完成：", OUTPUT_PATH)


if __name__ == "__main__":
    main()
