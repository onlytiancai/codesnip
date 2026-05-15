#!/usr/bin/env python3
"""
英语场景口语练习数据生成脚本
解析 docs/scenarios.md，随机选择场景和任务，生成中文句子及多条英文翻译

Features:
- 保存历史句子到文件，避免重复
- JSON 结果保存到编号子目录
- 生成新句子时参考同场景同任务的历史句子

Usage:
    python generate_sentence.py [--debug]
"""

import random
import json
import argparse
from pathlib import Path

# 导入本地模块
from markdown_parser import ScenarioParser, Scene, Task
from llm_caller import LLMCaller


# 路径配置
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
SCENARIOS_PATH = PROJECT_DIR / "docs" / "scenarios.md"
OUTPUT_DIR = SCRIPT_DIR / "output"
SENTENCES_FILE = OUTPUT_DIR / "sentences.txt"


# 系统提示词
SYSTEM_PROMPT = """你是一位专业的英语口语教练。用户会给你一个场景和任务，你需要：
1. 构思一个具体的上下文背景情境（用中文描述一个真实的对话场景，包含人物关系和具体情境）
2. 基于这个情境，生成一句常用的中文口语句子（不要直译，要符合真实场景需求）
3. 给出 3-5 种不同的英文表达方式
4. 每种表达要有：
   - sentence: 英文句子
   - style: 风格标签（polite 正式、neutral 中性、casual 口语化）
   - note: 简短说明
   - literal_translation: 中文直译，将英文逐词翻译成中文
   - phonetic: 音标，只标注元音和重音位置，简明易懂，便于用户朗读

请以 JSON 格式返回：
{
    "context": "上下文背景描述（中文）",
    "sentence_zh": "中文句子",
    "translations": [
        {"sentence": "英文句子", "style": "polite/neutral/casual", "note": "简短说明", "literal_translation": "中文直译", "phonetic": "音标"},
        ...
    ],
    "explanation": "整体讲解，说明这个句子适用的场景和注意事项"
}

只返回 JSON，不要有其他内容。"""


class SentenceData:
    """句子数据结构"""

    def __init__(self, scene: Scene, task: Task, data: dict):
        self.scene_en = scene["name_en"]
        self.scene_zh = scene["name_zh"]
        self.task_en = task["name_en"]
        self.task_zh = task["name_zh"]
        self.context = data.get("context", "")
        self.sentence_zh = data.get("sentence_zh", "")
        self.translations = data.get("translations", [])
        self.explanation = data.get("explanation", "")

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "scene_en": self.scene_en,
            "scene_zh": self.scene_zh,
            "task_en": self.task_en,
            "task_zh": self.task_zh,
            "context": self.context,
            "sentence_zh": self.sentence_zh,
            "translations": self.translations,
            "explanation": self.explanation
        }

    def print_result(self):
        """打印结果"""
        print(f"\n📍 场景：{self.scene_zh} ({self.scene_en})")
        print(f"📋 任务：{self.task_zh} ({self.task_en})")
        if self.context:
            print(f"📝 背景：{self.context}")
        print(f"💬 中文：{self.sentence_zh}\n")

        print("🌐 英文翻译：")
        for i, t in enumerate(self.translations, 1):
            print(f"  {i}. [{t['style']}] {t['sentence']}")
            if t.get('phonetic'):
                print(f"     音标：{t['phonetic']}")
            print(f"     直译：{t.get('literal_translation', '')}")
            print(f"     说明：{t['note']}")
            print()

        print(f"📖 讲解：{self.explanation}")


def parse_llm_response(content: list[dict], scene: Scene, task: Task) -> SentenceData:
    """解析 LLM 返回的内容"""
    for block in content:
        if block.type == "text":
            text = block.text.strip()

            # 清理 markdown 代码块
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

            try:
                data = json.loads(text)
                return SentenceData(scene, task, data)
            except json.JSONDecodeError as e:
                # 尝试修复不完整的 JSON
                last_brace = text.rfind('}')
                last_bracket = text.rfind(']')
                cutoff = max(last_brace, last_bracket)
                if cutoff > 0:
                    try:
                        data = json.loads(text[:cutoff + 1])
                        return SentenceData(scene, task, data)
                    except json.JSONDecodeError:
                        pass

                return SentenceData(scene, task, {
                    "sentence_zh": f"解析失败: {e}",
                    "translations": [],
                    "explanation": text[:500]
                })

    return SentenceData(scene, task, {
        "sentence_zh": "无响应",
        "translations": [],
        "explanation": ""
    })


def ensure_output_dir():
    """确保输出目录存在"""
    OUTPUT_DIR.mkdir(exist_ok=True)


def get_next_file_number() -> int:
    """获取下一个文件编号"""
    if not OUTPUT_DIR.exists():
        return 1
    existing = list(OUTPUT_DIR.glob("*.json"))
    if not existing:
        return 1
    # 找到最大编号
    max_num = 0
    for f in existing:
        try:
            num = int(f.stem)
            max_num = max(max_num, num)
        except ValueError:
            pass
    return max_num + 1


def save_json_result(result: SentenceData) -> int:
    """保存 JSON 结果到文件，返回使用的编号"""
    ensure_output_dir()
    file_num = get_next_file_number()
    filepath = OUTPUT_DIR / f"{file_num}.json"
    filepath.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\n💾 已保存 JSON 到：{filepath}")
    return file_num


def append_sentence_to_history(result: SentenceData, file_num: int):
    """追加句子到历史记录文件"""
    ensure_output_dir()

    # 每行格式：序号 | 场景 | 任务 | 中文句子（用 | 分割避免逗号冲突）
    line = f"{file_num} | {result.scene_zh} | {result.task_zh} | {result.sentence_zh}\n"

    with open(SENTENCES_FILE, "a", encoding='utf-8') as f:
        f.write(line)

    print(f"📝 已追加到历史记录：{SENTENCES_FILE}")


def load_historical_sentences(scene: Scene, task: Task) -> list[str]:
    """加载同一场景同一任务的历史句子"""
    if not SENTENCES_FILE.exists():
        return []

    content = SENTENCES_FILE.read_text(encoding='utf-8')
    sentences = []

    # 每行格式：序号 | 场景 | 任务 | 中文句子
    for line in content.split('\n'):
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split('|')]

        if len(parts) != 4:
            continue

        scene_name = parts[1]
        task_name = parts[2]
        sentence = parts[3]

        if scene_name == scene['name_zh'] and task_name == task['name_zh']:
            sentences.append(sentence)

    return sentences


def build_prompt_with_history(scene: Scene, task: Task, historical: list[str]) -> str:
    """构建带有历史记录的 prompt"""
    prompt = f"""场景：{scene['name_zh']} ({scene['name_en']})
任务：{task['name_zh']} ({task['name_en']})
"""

    if historical:
        prompt += f"\n该场景任务下已有的中文句子（请勿重复）：\n"
        for i, s in enumerate(historical, 1):
            prompt += f"{i}. {s}\n"
        prompt += "\n请生成一句与以上句子不重复的中文口语句子。\n"
    else:
        prompt += "请为这个任务生成一句常用的中文口语句子。\n"

    prompt += "\n请给出多种英文表达方式。"

    return prompt


def main():
    parser = argparse.ArgumentParser(description="英语场景口语练习数据生成")
    parser.add_argument("--debug", action="store_true", help="启用 debug 模式")
    args = parser.parse_args()

    # 解析 scenarios.md
    scenario_parser = ScenarioParser.from_file(SCENARIOS_PATH)
    scenes = scenario_parser.parse()

    print(f"📚 解析到 {len(scenes)} 个场景\n")

    # 随机选择一个场景和任务
    scene = random.choice(scenes)
    task = random.choice(scene["tasks"])

    print(f"🎲 随机选择：场景「{scene['name_zh']}」任务「{task['name_zh']}」")

    # 加载历史句子
    historical = load_historical_sentences(scene, task)
    if historical:
        print(f"📜 该场景任务已有 {len(historical)} 条历史句子，将用于去重")

    # 构建 prompt
    prompt = build_prompt_with_history(scene, task, historical)

    # 调用 LLM
    caller = LLMCaller(debug=args.debug)
    content = caller.call(SYSTEM_PROMPT, prompt)

    # 解析结果
    result = parse_llm_response(content, scene, task)
    result.print_result()

    # 保存结果
    file_num = save_json_result(result)
    append_sentence_to_history(result, file_num)


if __name__ == "__main__":
    main()