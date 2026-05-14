#!/usr/bin/env python3
"""
英语场景口语练习数据生成脚本
解析 docs/scenarios.md，随机选择场景和任务，生成中文句子及多条英文翻译

Usage:
    python generate_sentence.py [--debug]
"""

import random
import json
import argparse
from pathlib import Path

# 导入本地模块
from markdown_parser import ScenarioParser, Scene, Task
from llm_caller import LLMCaller, call_llm


# 路径配置
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
SCENARIOS_PATH = PROJECT_DIR / "docs" / "scenarios.md"

# 系统提示词
SYSTEM_PROMPT = """你是一位专业的英语口语教练。用户会给你一个场景和任务，你需要：
1. 构思一句符合该场景和任务的常用中文口语句子（不要直译，要符合真实场景需求）
2. 给出 3-5 种不同的英文表达方式
3. 每种表达要有：
   - style：风格标签（polite 正式、neutral 中性、casual 口语化）
   - note：简短说明
   - literal_translation：中文直译，将英文逐词翻译成中文
   - keywords：重点单词列表，每个包含 word（单词）、phonetic（音标）、translation（中文翻译）

请以 JSON 格式返回：
{
    "sentence_zh": "中文句子",
    "translations": [
        {"sentence": "英文句子", "style": "polite/neutral/casual", "note": "简短说明", "literal_translation": "中文直译", "keywords": [{"word": "单词", "phonetic": "音标", "translation": "中文翻译"}, ...]},
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
        self.sentence_zh = data.get("sentence_zh", "")
        self.translations = data.get("translations", [])
        self.explanation = data.get("explanation", "")

    def print_result(self):
        """打印结果"""
        print(f"\n📍 场景：{self.scene_zh} ({self.scene_en})")
        print(f"📋 任务：{self.task_zh} ({self.task_en})")
        print(f"💬 中文：{self.sentence_zh}\n")

        print("🌐 英文翻译：")
        for i, t in enumerate(self.translations, 1):
            print(f"  {i}. [{t['style']}] {t['sentence']}")
            print(f"     直译：{t.get('literal_translation', '')}")
            print(f"     说明：{t['note']}")
            if "keywords" in t and t["keywords"]:
                keywords = ", ".join(
                    f"{kw['word']} {kw['phonetic']} ({kw['translation']})"
                    for kw in t["keywords"]
                )
                print(f"     重点：{keywords}")
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
                # 查找最后一个完整的对象
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

    # 调用 LLM
    prompt = f"""场景：{scene['name_zh']} ({scene['name_en']})
任务：{task['name_zh']} ({task['name_en']})

请为这个任务生成一句常用的中文口语句子，并给出多种英文表达方式。"""

    caller = LLMCaller(debug=args.debug)
    content = caller.call(SYSTEM_PROMPT, prompt)

    # 解析结果
    result = parse_llm_response(content, scene, task)
    result.print_result()

    # 如果有 debug 信息，也打印出来
    if args.debug and caller.get_debug_info():
        # debug 信息已在 LLMCaller.call() 中打印
        pass


if __name__ == "__main__":
    main()