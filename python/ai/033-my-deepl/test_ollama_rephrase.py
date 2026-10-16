"""批量跑几个中→英 → ollama_rephrase 的例子,看模型在小模型上的稳定性。

跑法::

    ~/.pyenv/versions/3.11.9/bin/python3.11 test_ollama_rephrase.py

每条用例打印:
    [CASE n] source / current / target
    [OUTPUT]   6 条候选(或少于 6)
    [CHECKS]   通过/不通过的硬约束 (前缀保留、head≠target、去重)
    [SOFT]     主观观察(是否包含"语义上"接近 target 的近义改写、是否完全照抄)
"""
from __future__ import annotations

import json
import sys

from ollama_rephrase import rephrase_candidates


CASES: list[dict] = [
    # 0. README 原例,目标位置在第 6 个词(中间)。
    {
        "source": "您的使用反馈,将影响我们下一步的优化方向",
        "current": "Your feedback from using it will shape the direction of our next round of improvements.",
        "prefix": "Your feedback from using it will",
        "target": "shape",
    },
    # 1. 短句,目标词是高频 "is"
    {
        "source": "埃隆·马斯克不在领英上。",
        "current": "Elon Musk is not on LinkedIn.",
        "prefix": "Elon Musk",
        "target": "is",
    },
    # 2. 复合句,目标词在末尾
    {
        "source": "请把灯关上,因为我要睡觉了。",
        "current": "Please turn off the lights, because I am going to sleep.",
        "prefix": "Please turn off the lights, because I am going to",
        "target": "sleep",
    },
    # 3. 短句,目标词在中间
    {
        "source": "今天天气很好。",
        "current": "Today's weather is really nice.",
        "prefix": "Today's weather",
        "target": "is",
    },
    # 4. 较长的复合句,目标词接近句首
    {
        "source": "尽管他很努力,他仍然没有通过考试。",
        "current": "Although he tried hard, he still did not pass the exam.",
        "prefix": "Although he tried hard, he still did not",
        "target": "pass",
    },
    # 5. 高频动词 "use",中段位置
    {
        "source": "我们使用 Python 来处理数据。",
        "current": "We use Python to process the data.",
        "prefix": "We",
        "target": "use",
    },
    # 6. 抽象名词 "idea"
    {
        "source": "这个想法值得讨论。",
        "current": "This idea deserves discussion.",
        "prefix": "This",
        "target": "idea",
    },
    # 7. 形容词 "important"
    {
        "source": "数据安全对公司来说很重要。",
        "current": "Data security is very important to the company.",
        "prefix": "Data security is very",
        "target": "important",
    },
    # 8. 中段动词,prefix 短,留给模型发挥空间大
    {
        "source": "他昨天买了一辆新车。",
        "current": "He bought a brand new car yesterday.",
        "prefix": "He",
        "target": "bought",
    },
    # 9. 句首动词
    {
        "source": "参加这次会议的有二十多人。",
        "current": "More than twenty people attended the meeting.",
        "prefix": "More than twenty people",
        "target": "attended",
    },
]


def _check(cands: list[str], *, current: str, prefix: str, target: str) -> tuple[list[str], list[str]]:
    """硬约束检查。返回 (失败列表, 警告列表)。"""
    fails: list[str] = []
    warns: list[str] = []
    pfx_lc = prefix.strip().lower().rstrip(" .!?")
    target_lc = target.lower().rstrip(" .,!?;:")
    cur_lc = current.strip().lower().rstrip(" .!?")
    seen: set[str] = set()
    for i, c in enumerate(cands, 1):
        cl = c.lower().rstrip(" .!?")
        if not cl.startswith(pfx_lc):
            fails.append(f"#{i} 不以 prefix 开头: {c!r}")
        # 紧跟 prefix 之后的 head
        rest = c[len(prefix):].lstrip() if c.lower().startswith(pfx_lc) else c
        head = rest.split(maxsplit=1)[0].lower().rstrip(" .,!?;:") if rest else ""
        if head == target_lc:
            fails.append(f"#{i} head 词 == target: {c!r}")
        if cl == cur_lc:
            fails.append(f"#{i} 照抄 current: {c!r}")
        if cl in seen:
            fails.append(f"#{i} 重复候选: {c!r}")
        seen.add(cl)
    if not cands:
        fails.append("零候选")
    return fails, warns


def main() -> int:
    n = 6
    overall_fails = 0
    for idx, case in enumerate(CASES):
        print(f"\n========== CASE {idx} ==========")
        print("source :", case["source"])
        print("current:", case["current"])
        print("prefix :", repr(case["prefix"]))
        print("target :", repr(case["target"]))
        try:
            cands = rephrase_candidates(
                source_zh=case["source"],
                current=case["current"],
                prefix=case["prefix"],
                target_word=case["target"],
                n=n,
            )
        except Exception as e:  # noqa: BLE001
            print(f"  EXCEPTION: {e!r}")
            overall_fails += 1
            continue
        if not cands:
            print("  (no candidates)")
            overall_fails += 1
            continue
        print(f"  -- {len(cands)} candidates --")
        for i, c in enumerate(cands, 1):
            print(f"  {i}. {c}")
        fails, warns = _check(cands, current=case["current"], prefix=case["prefix"], target=case["target"])
        if fails:
            overall_fails += 1
            for f in fails:
                print(f"  [FAIL] {f}")
        for w in warns:
            print(f"  [WARN] {w}")
    print(f"\n========== TOTAL FAIL CASES: {overall_fails}/{len(CASES)} ==========")
    return 0 if overall_fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
