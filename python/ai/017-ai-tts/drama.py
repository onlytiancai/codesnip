#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
drama.py — 小说 → 多人配音有声剧 流水线（阶段一：analyze）。

子命令：
  analyze   读小说文本，调 MiniMax LLM 产出剧本 JSON（仅分析，不合成音频）

后续阶段（暂未实现）：
  synth     按 script.json 逐句调 MiniMax TTS，产出 wav 片段
  merge     用 pydub 把片段按 pause_after_ms 拼接为最终有声剧
  all       analyze + synth + merge 一条龙
"""

import argparse
import json
import os
import re
import sys

import anthropic

import voices

# ---- LLM 常量（走 Anthropic SDK，"MiniMax-M3" 是模型名） ----
DEFAULT_MODEL = "MiniMax-M3"

# TTS 模型 + 语言增强（写入 script.json 给后续 synth 阶段用）
DEFAULT_TTS_MODEL = "speech-02-hd"
DEFAULT_LANG_BOOST = "Chinese"

# ---- emotion 枚举（TTS 接口限定）----
VALID_EMOTIONS = {
    "happy", "sad", "angry", "fearful", "disgusted",
    "surprised", "calm", "fluent", "whisper",
}


# ===================== 工具 =====================

def _strip_think(text):
    """剥离 MiniMax-M3 可能输出的 <think>…</think> 推理段。"""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_json(text):
    """从模型输出中提取首个完整的 {…} JSON 块。"""
    text = _strip_think(text)
    # 先尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 退化：扫描首个 { 到匹配的 }
    start = text.find("{")
    if start == -1:
        raise ValueError("模型输出中未找到 JSON 对象")
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("模型输出中 JSON 大括号未闭合")


def _post_chat(system, messages, model=DEFAULT_MODEL, max_tokens=8192):
    """调 Anthropic SDK（模型名 MiniMax-M3），返回拼好的 text 字符串。

    - system: 字符串，独立传入 SDK 的 system 字段
    - messages: [{role: "user"|"assistant", content: [...] | str}, ...]
    - 返回：把所有 type=="text" 的 block.text 拼起来
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "环境变量 ANTHROPIC_API_KEY 未设置。\n"
            "  export ANTHROPIC_API_KEY=...\n"
            "若网络不通可挂代理：export HTTPS_PROXY=http://127.0.0.1:10808"
        )

    client = anthropic.Anthropic()
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=messages,
    )

    parts = []
    for block in message.content:
        # 推理模型会同时有 type=="thinking" 的 block，按参考脚本那样跳过
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "\n".join(parts).strip()


def _normalize_emotion(e):
    """越界 emotion 回落 calm。"""
    if not isinstance(e, str):
        return "calm"
    e = e.strip().lower()
    return e if e in VALID_EMOTIONS else "calm"


def _normalize_role_block(role_dict):
    """规范化角色块，确保必要字段存在。"""
    out = {
        "voice_id": role_dict.get("voice_id", "audiobook_female_1"),
        "speed":   float(role_dict.get("speed", 1.0)),
        "vol":     float(role_dict.get("vol", 1.0)),
        "pitch":   int(role_dict.get("pitch", 0)),
        "desc":    str(role_dict.get("desc", "")),
    }
    # 越界修正
    out["speed"] = max(0.5, min(2.0, out["speed"]))
    out["vol"]   = max(0.0, min(10.0, out["vol"]))
    out["pitch"] = max(-12, min(12, out["pitch"]))
    return out


def _normalize_line(line, idx):
    """规范化台词行。"""
    return {
        "idx":             int(line.get("idx", idx)),
        "role":            str(line.get("role", "旁白")),
        "text":            str(line.get("text", "")),
        "emotion":         _normalize_emotion(line.get("emotion", "calm")),
        "vol":             float(line.get("vol", 1.0)),
        "speed":           float(line.get("speed", 1.0)),
        "pitch":           int(line.get("pitch", 0)),
        "pause_after_ms":  int(line.get("pause_after_ms", 300)),
        "pronunciation":   list(line.get("pronunciation", []) or []),
    }


def _normalize_script(raw, tts_model, lang_boost, fallback_title="未命名剧本"):
    """规范化整本剧本，校验必要字段、补默认。"""
    if not isinstance(raw, dict):
        raise ValueError("模型输出顶层不是对象")

    title = str(raw.get("title", fallback_title))
    roles_raw = raw.get("roles", {}) or {}
    lines_raw = raw.get("lines", []) or []

    roles = {str(k): _normalize_role_block(v) for k, v in roles_raw.items()}
    lines = [_normalize_line(line, i) for i, line in enumerate(lines_raw)]

    return {
        "title":          title,
        "tts_model":      str(raw.get("tts_model", tts_model)),
        "language_boost": str(raw.get("language_boost", lang_boost)),
        "roles":          roles,
        "lines":          lines,
    }


# ===================== Prompt =====================

SYSTEM_PROMPT = """你是一位中文有声剧编剧，擅长把武侠/言情/历史小说改编成多人配音剧本。

【任务】
阅读用户提供的【小说原文】，把它拆解成可逐句配音的结构化 JSON。

【输出格式（严格遵守）】
只输出一个 JSON 对象，不要任何解释/前缀/后缀，禁止使用 markdown 代码块包裹。
JSON 结构如下：

{
  "title": "本段标题（自拟，简洁）",
  "tts_model": "speech-02-hd",
  "language_boost": "Chinese",
  "roles": {
    "<角色名>": {
      "voice_id": "<见音色表>",
      "speed": 1.0,
      "vol": 1.0,
      "pitch": 0,
      "desc": "<一句话人设>"
    }
  },
  "lines": [
    {
      "idx": 0,
      "role": "<角色名>",
      "text": "<这一句原文台词或叙述>",
      "emotion": "<happy|sad|angry|fearful|disgusted|surprised|calm|fluent|whisper>",
      "vol": 1.0,
      "speed": 1.0,
      "pitch": 0,
      "pause_after_ms": 300,
      "pronunciation": []
    }
  ]
}

【角色与台词拆分规则】
1. 角色清单必须包含「旁白」（叙述者），名字固定为 "旁白"。
2. 旁白负责：场景描写、人物动作、心理活动、不属于任何对白的叙述。
3. 对话按书中的引号内容逐句拆分，每句一个 lines 条目；同一角色连续多句也要拆开，方便后期调整。
4. 若某角色只出场无对白（如"青衣少女"），仍要写入 roles 给个音色，但不要给他台词。
5. 角色名用原文称呼（林平之、郑镖头、萨老头、店主人老蔡 等）。

【emotion 枚举（必须严格匹配）】
happy / sad / angry / fearful / disgusted / surprised / calm / fluent / whisper
- 旁白叙述默认 calm；激烈打斗可用 angry；温情感人用 warm(→calm)；调侃用 happy。
- 给出 emotion 后若不在枚举里，程序会强制回落 calm，所以请直接给枚举值。

【vol / speed / pitch】
- vol: 0–10，默认 1.0；平静场景默认；激烈/高喊可适当提高。
- speed: 0.5–2.0，默认 1.0；老者可降到 0.9；少女/快嘴可升到 1.1。
- pitch: -12–12，默认 0；少年 +1~+3；老者 -2~-4；女性 +2；男性 -2 等。

【pause_after_ms】
本句播完后的停顿（毫秒）。默认 300；句号/段落结束 400~600；场景切换 600~800；
紧张对峙 200；笑声、惊讶 150。

【pronunciation 多音字】
遇到容易读错的多音字时填字符串列表，例如 ["处理/(chu3)(li3)"]；
没有就填空列表 []。

【音色表 voice_id（务必从这里选）】
""" + voices.list_voices() + """

【选角原则】
- 旁白：用 audiobook_female_1 / audiobook_male_1 / presenter_female 等叙述音色。
- 少年公子：male-qn-qingse / male-qn-yangguang。
- 精英/镖师/中年职业男性：male-qn-jingying / male-qn-zhixing / male-qn-chengshu。
- 霸道/反派：male-qn-badao。
- 温柔女主：female-wenrou / female-tianmei。
- 少女/丫鬟：female-shaonv。
- 御姐/成熟女主：female-yujie / female-chengshu。
- 老者/乡土老人：audiobook_male_1 或 male-qn-chengshu，speed 降到 0.9，pitch 降到 -2。

【再次强调】只输出 JSON，不要任何额外文字（包括 <think>）。"""


# ===================== analyze 子命令 =====================

def cmd_analyze(args):
    if not os.path.isfile(args.input):
        sys.exit(f"找不到输入文件: {args.input}")

    with open(args.input, "r", encoding="utf-8") as f:
        novel_text = f.read().strip()

    if not novel_text:
        sys.exit(f"输入文件为空: {args.input}")

    messages = [
        {"role": "user", "content": f"【小说原文】\n{novel_text}"},
    ]

    print(f"[analyze] 调用 LLM ({args.model})，原文 {len(novel_text)} 字…", file=sys.stderr)
    content = _post_chat(
        system=SYSTEM_PROMPT,
        messages=messages,
        model=args.model,
        max_tokens=args.max_tokens,
    )

    # 调试可打开
    if args.verbose:
        print("---- 模型原始输出（前 500 字）----", file=sys.stderr)
        print(content[:500], file=sys.stderr)
        print("---- end ----", file=sys.stderr)

    raw = _extract_json(content)
    script = _normalize_script(
        raw,
        tts_model=args.tts_model,
        lang_boost=args.lang_boost,
        fallback_title=os.path.splitext(os.path.basename(args.input))[0],
    )

    # 落盘
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(script, f, ensure_ascii=False, indent=2)

    # 摘要
    print(f"[analyze] 已写入 {args.output}")
    print(f"  标题: {script['title']}")
    print(f"  角色 ({len(script['roles'])}):")
    for r, meta in script["roles"].items():
        print(f"    - {r}: voice_id={meta['voice_id']} ({meta['desc']})")
    print(f"  台词行数: {len(script['lines'])}")
    by_role = {}
    for line in script["lines"]:
        by_role[line["role"]] = by_role.get(line["role"], 0) + 1
    print("  各角色台词分布:")
    for r, n in by_role.items():
        print(f"    - {r}: {n} 句")


# ===================== 入口 =====================

def main():
    parser = argparse.ArgumentParser(
        description="小说 → 多人配音有声剧（MiniMax）"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_an = sub.add_parser("analyze", help="读文本，调 LLM 产出剧本 JSON")
    p_an.add_argument("--input",  "-i", required=True, help="小说原文 .txt")
    p_an.add_argument("--output", "-o", required=True, help="输出 script.json")
    p_an.add_argument("--model",         default=DEFAULT_MODEL,        help=f"LLM 模型（默认 {DEFAULT_MODEL}）")
    p_an.add_argument("--tts-model",     default=DEFAULT_TTS_MODEL,    help=f"写入 JSON 的 TTS 模型（默认 {DEFAULT_TTS_MODEL}）")
    p_an.add_argument("--lang-boost",    default=DEFAULT_LANG_BOOST,   help=f"语言增强（默认 {DEFAULT_LANG_BOOST}）")
    p_an.add_argument("--max-tokens",    type=int, default=8192,        help="LLM 最大输出 token")
    p_an.add_argument("--verbose", "-v", action="store_true",         help="打印模型原始输出（调试用）")
    p_an.set_defaults(func=cmd_analyze)

    # 后续子命令占位（synth / merge / all）暂不实现
    for name in ("synth", "merge", "all"):
        p = sub.add_parser(name, help="[未实现] 后续阶段")
        p.set_defaults(func=lambda a, _n=name: sys.exit(f"[{_n}] 尚未实现，请先 review analyze 产出"))

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()