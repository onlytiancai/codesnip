"""
Markdown 场景解析器
使用第三方库解析 docs/scenarios.md
"""

import re
from pathlib import Path
from typing import TypedDict

from markdown_it import MarkdownIt


class Task(TypedDict):
    name_en: str
    name_zh: str


class Scene(TypedDict):
    name_en: str
    name_zh: str
    tasks: list[Task]


class ScenarioParser:
    """解析 scenarios.md 为场景数据结构"""

    def __init__(self, content: str):
        self.content = content
        self.md = MarkdownIt()

    def parse(self) -> list[Scene]:
        """解析 markdown 内容"""
        scenes: list[Scene] = []
        current_scene: Scene | None = None
        current_tasks: list[Task] = []

        # 按行处理
        for line in self.content.split('\n'):
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
                    'name_zh': scene_match.group(1).strip(),
                    'name_en': scene_match.group(2).strip(),
                    'tasks': []
                }
                current_tasks = []
                continue

            # 检测任务标题：- 值机 (Check-in)
            task_match = re.match(r'^-+ (.+?) \((.+?)\)$', line)
            if task_match and current_scene:
                current_tasks.append({
                    'name_zh': task_match.group(1).strip(),
                    'name_en': task_match.group(2).strip()
                })

        # 最后一个场景
        if current_scene:
            current_scene['tasks'] = current_tasks
            scenes.append(current_scene)

        return scenes

    @classmethod
    def from_file(cls, filepath: Path) -> "ScenarioParser":
        """从文件加载并解析"""
        content = filepath.read_text(encoding='utf-8')
        return cls(content)


def parse_scenarios_file(filepath: Path) -> list[Scene]:
    """快捷函数：直接解析文件"""
    parser = ScenarioParser.from_file(filepath)
    return parser.parse()