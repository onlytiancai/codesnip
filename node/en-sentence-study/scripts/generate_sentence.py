"""
英语场景口语练习数据生成脚本
解析 docs/scenarios.md，随机选择场景和任务，生成中文句子及多条英文翻译
"""

import random
import re
import json
from pathlib import Path
from typing import TypedDict

import anthropic


class Task(TypedDict):
    name_en: str
    name_zh: str


class Scene(TypedDict):
    name_en: str
    name_zh: str
    tasks: list[Task]


class SentenceData(TypedDict):
    scene_en: str
    scene_zh: str
    task_en: str
    task_zh: str
    sentence_zh: str
    translations: list[dict]
    explanation: str


SCENARIOS_PATH = Path(__file__).parent.parent / "docs" / "scenarios.md"
SYSTEM_PROMPT = """你是一位专业的英语口语教练。用户会给你一个场景和任务，你需要：
1. 构思一句符合该场景和任务的常用中文口语句子（不要直译，要符合真实场景需求）
2. 给出 3-5 种不同的英文表达方式
3. 每种表达要有风格标签：polite（正式）、neutral（中性）、casual（口语化）
4. 每种表达要有简短说明

请以 JSON 格式返回：
{
    "sentence_zh": "中文句子",
    "translations": [
        {"sentence": "英文句子", "style": "polite/neutral/casual", "note": "简短说明"},
        ...
    ],
    "explanation": "整体讲解，说明这个句子适用的场景和注意事项"
}

只返回 JSON，不要有其他内容。"""


def parse_scenarios(content: str) -> list[Scene]:
    """解析 markdown 内容为场景数据结构"""
    scenes: list[Scene] = []
    current_scene: Scene | None = None
    current_tasks: list[Task] = []

    for line in content.split('\n'):
        line = line.rstrip()
        if not line:
            continue

        # 检测场景标题：## 机场 (Airport)
        scene_match = re.match(r'^## (.+?) \((.+?)\)$', line)
        if scene_match:
            if current_scene:
                current_scene['tasks'] = current_tasks
                scenes.append(current_scene)
            current_scene = {
                'name_zh': scene_match.group(1),
                'name_en': scene_match.group(2),
                'tasks': []
            }
            current_tasks = []
            continue

        # 检测任务标题：- 值机 (Check-in)
        task_match = re.match(r'^-+ (.+?) \((.+?)\)$', line)
        if task_match and current_scene:
            current_tasks.append({
                'name_zh': task_match.group(1),
                'name_en': task_match.group(2)
            })

    if current_scene:
        current_scene['tasks'] = current_tasks
        scenes.append(current_scene)

    return scenes


def generate_sentence(scene: Scene, task: Task, client: anthropic.Anthropic) -> SentenceData:
    """调用 LLM API 生成句子和翻译"""
    prompt = f"""场景：{scene['name_zh']} ({scene['name_en']})
任务：{task['name_zh']} ({task['name_en']})

请为这个任务生成一句常用的中文口语句子，并给出多种英文表达方式。"""

    message = client.messages.create(
        model="MiniMax-M2.7",
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }]
    )

    # 打印 LLM 原始返回
    for block in message.content:
        if block.type == "text":
            print(f"=== LLM 原始返回 ===\n{block.text}\n=== END ===\n")

            # 清理可能的 markdown 代码块
            text = block.text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

            try:
                data = json.loads(text)
                return {
                    "scene_en": scene["name_en"],
                    "scene_zh": scene["name_zh"],
                    "task_en": task["name_en"],
                    "task_zh": task["name_zh"],
                    "sentence_zh": data.get("sentence_zh", ""),
                    "translations": data.get("translations", []),
                    "explanation": data.get("explanation", "")
                }
            except json.JSONDecodeError as e:
                return {
                    "scene_en": scene["name_en"],
                    "scene_zh": scene["name_zh"],
                    "task_en": task["name_en"],
                    "task_zh": task["name_zh"],
                    "sentence_zh": f"解析失败: {e}",
                    "translations": [],
                    "explanation": text
                }

    return {
        "scene_en": scene["name_en"],
        "scene_zh": scene["name_zh"],
        "task_en": task["name_en"],
        "task_zh": task["name_zh"],
        "sentence_zh": "",
        "translations": [],
        "explanation": "无响应"
    }


def main():
    # 解析 scenarios.md
    content = SCENARIOS_PATH.read_text(encoding='utf-8')
    scenes = parse_scenarios(content)

    print(f"解析到 {len(scenes)} 个场景\n")

    # 随机选择一个场景和任务
    scene = random.choice(scenes)
    task = random.choice(scene['tasks'])

    print(f"场景：{scene['name_zh']} ({scene['name_en']})")
    print(f"任务：{task['name_zh']} ({task['name_en']})\n")

    # 调用 LLM 生成
    client = anthropic.Anthropic()
    result = generate_sentence(scene, task, client)

    print(f"中文句子：{result['sentence_zh']}\n")
    print("英文翻译：")
    for t in result['translations']:
        print(f"  [{t['style']}] {t['sentence']}")
        print(f"         说明：{t['note']}\n")
    print(f"讲解：{result['explanation']}")


if __name__ == "__main__":
    main()