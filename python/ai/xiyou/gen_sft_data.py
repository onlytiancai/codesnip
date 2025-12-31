import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ====== 配置区 ======
MODEL_PATH = "/Users/huhao/models/Qwen2.5-3B-Instruct"
TEXT_PATH = "./data/xiyou_raw.txt"
OUTPUT_PATH = "xiyou_instruct.jsonl"

DEVICE = "mps"
MAX_NEW_TOKENS = 512
CHUNK_SIZE = 600   # 每段字符数，建议 400~800
STRIDE = 100       # 滑动窗口，防止断章
# ====================


def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    # 简单清洗
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
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    text = load_text(TEXT_PATH)
    chunks = split_text(text)

    print(f"共切分得到 {len(chunks)} 段文本")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
        for idx, chunk in enumerate(chunks):
            prompt = build_prompt(chunk)
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )

            decoded = tokenizer.decode(
                outputs[0], skip_special_tokens=True
            )

            qa_pairs = parse_qa(decoded)

            for q, a in qa_pairs:
                record = {
                    "instruction": q,
                    "input": "",
                    "output": a
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            if idx % 10 == 0:
                print(f"已处理 {idx}/{len(chunks)} 段")

    print("数据生成完成，输出文件：", OUTPUT_PATH)


if __name__ == "__main__":
    main()
