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
import urllib.request
import urllib.error

import anthropic

import voices

# ---- LLM 常量（走 Anthropic SDK，"MiniMax-M3" 是模型名） ----
DEFAULT_MODEL = "MiniMax-M3"

# ---- TTS 常量（MiniMax /v1/t2a_v2） ----
TTS_URL = "https://api.minimaxi.com/v1/t2a_v2"

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


# ===================== TTS =====================

def _clamp_speed(v):  return max(0.5, min(2.0, float(v)))
def _clamp_vol(v):    return max(0.0, min(10.0, float(v)))
def _clamp_pitch(v):  return max(-12, min(12, int(v)))


def _decode_audio(s):
    """嗅探 hex / base64 编码的音频字符串，返回 bytes。仿 minimax-tts.ts。"""
    if re.fullmatch(r"[0-9a-fA-F]+", s) and len(s) % 2 == 0:
        return bytes.fromhex(s)
    import base64
    return base64.b64decode(s)


def _post_tts(text, voice_id, emotion="calm", vol=1.0, speed=1.0, pitch=0,
              pronunciation=None, tts_model=DEFAULT_TTS_MODEL,
              lang_boost=DEFAULT_LANG_BOOST, timeout=120, audio_format="mp3"):
    """调 MiniMax /v1/t2a_v2，返回 dict：
       {audio_bytes, audio_length_ms, sample_rate, channels, format, usage_chars}。

    pronunciation: list[str]，例如 ["处理/(chu3)(li3)"]，空列表表示不传。
    """
    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        raise RuntimeError(
            "环境变量 MINIMAX_API_KEY 未设置（TTS 需要，与 ANTHROPIC_API_KEY 不同）。\n"
            "  export MINIMAX_API_KEY=...\n"
            "若网络不通可挂代理：export HTTPS_PROXY=http://127.0.0.1:10808"
        )

    pronunciation = list(pronunciation or [])
    payload = {
        "model":          tts_model,
        "text":           text,
        "stream":         False,
        "language_boost": lang_boost,
        "voice_setting": {
            "voice_id": voice_id,
            "speed":    _clamp_speed(speed),
            "vol":      _clamp_vol(vol),
            "pitch":    _clamp_pitch(pitch),
            "emotion":  _normalize_emotion(emotion),
        },
        "audio_setting": {
            "sample_rate": 32000,
            "bitrate":      128000,
            "format":       audio_format,
            "channel":      1,
        },
    }
    if pronunciation:
        payload["pronunciation_dict"] = {"tone": pronunciation}

    req = urllib.request.Request(
        TTS_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"MiniMax TTS HTTP {e.code}: {err}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"无法连接 MiniMax TTS（{e.reason}）。"
            f"若需代理：export HTTPS_PROXY=http://127.0.0.1:10808"
        ) from e

    base = body.get("base_resp", {}) or {}
    if base.get("status_code", 0) != 0:
        raise RuntimeError(
            f"MiniMax TTS 返回错误：{base.get('status_msg')} (code={base.get('status_code')})"
        )

    data = body.get("data", {}) or {}
    extra = body.get("extra_info", {}) or {}

    audio_str = data.get("audio")
    if not audio_str:
        raise RuntimeError(f"MiniMax TTS 响应缺少 data.audio：{body}")

    return {
        "audio_bytes":     _decode_audio(audio_str),
        "audio_length_ms": int(extra.get("audio_length", 0)),   # 单位：毫秒
        "sample_rate":     int(extra.get("audio_sample_rate", 32000)),
        "channels":        int(extra.get("audio_channel", 1)),
        "format":          extra.get("audio_format", audio_format),
        "usage_chars":     int(extra.get("usage_characters", len(text))),
    }


def _merge_line_with_role(line, role):
    """合并行级与角色级默认值（行级缺省时回退到角色级）。"""
    return {
        "voice_id":      role.get("voice_id", "audiobook_female_1"),
        "emotion":       _normalize_emotion(line.get("emotion", "calm")),
        "vol":           float(line.get("vol",     role.get("vol",   1.0))),
        "speed":         float(line.get("speed",   role.get("speed", 1.0))),
        "pitch":         int(  line.get("pitch",   role.get("pitch", 0))),
        "text":          str(  line.get("text", "")),
        "pause_after_ms":int(  line.get("pause_after_ms", 300)),
        "pronunciation": list( line.get("pronunciation", []) or []),
    }


def _normalize_emotion(e):
    """越界 emotion 回落 calm。"""
    if not isinstance(e, str):
        return "calm"
    e = e.strip().lower()
    return e if e in VALID_EMOTIONS else "calm"


def _normalize_role_block(role_dict):
    """规范化角色块，确保必要字段存在，并对 voice_id 做校验。"""
    desc = str(role_dict.get("desc", ""))
    voice_id = role_dict.get("voice_id", "audiobook_female_1")
    # voice_id 不在 voices.VOICES 时，按 desc 里的性别关键词选回退
    if voice_id not in voices.VOICES:
        d = desc
        is_female = any(k in d for k in ("女", "少女", "御姐", "姑娘", "少妇", "妹"))
        gender = voices.GENDER_FEMALE if is_female else voices.GENDER_MALE
        voice_id = voices.fallback_for(voice_id, gender=gender)
    out = {
        "voice_id": voice_id,
        "speed":   float(role_dict.get("speed", 1.0)),
        "vol":     float(role_dict.get("vol", 1.0)),
        "pitch":   int(role_dict.get("pitch", 0)),
        "desc":    desc,
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

【attribution + 引号 的硬规则 — 必须严格遵守】
原文里引号结构通常是：`郑镖头道："少镖头，咱们去喝一杯怎么样？"林平之笑道："你跟我出来打猎是假，喝酒才是正经事。"一勒马，飘身跃下马背。`

拆法（关键）：
  - 旁白行：[场景到冒号前] + 「道："」（attribution + 开引号）
    例：`旁白: "郑镖头道："`    ← 包含 attribution 和开引号
  - 说话人行：引号内的对白正文（不含引号本身）
    例：`郑镖头: "少镖头，咱们去喝一杯怎么样？"`
  - 旁白行：[闭引号之后到下一个 attribution 之前] 的所有叙述
    例：`旁白: '"林平之笑道："'` ← 包含闭引号 + 下个 attribution + 开引号
    例：`旁白: "一勒马，飘身跃下马背。"`

严禁：
  - 丢 attribution（不要把 `郑镖头道："` 缩成只剩对白；`道/说道/笑道/叫道/答道/喝道/喊道/答道` 后面必须紧跟 `："`）
  - 把闭引号之后的叙述并入说话人行（如 `客官请坐，喝酒么？说的是北方口音。` → 应拆成萨老头说 `客官请坐，喝酒么？` + 旁白 `说的是北方口音。`）
  - 把叙述词（`这么奉承一番` / `说的是` / `心想` / `便道` / `当下` / `乃` / `总是`）塞进对白行
  - 漏抄原文（最终所有 line.text 拼起来应能覆盖原文每个非空白字符；attribution 动词、引号、句末标点全要保留）

英文/中文标点保留在 text 里（逗号、句号、引号、问号、感叹号、省略号）以体现语速与情感。

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


# ===================== synth 子命令 =====================

def cmd_synth(args):
    if not os.path.isfile(args.script):
        sys.exit(f"找不到剧本文件: {args.script}")
    with open(args.script, "r", encoding="utf-8") as f:
        script = json.load(f)

    lines  = script.get("lines", []) or []
    roles  = script.get("roles", {}) or {}
    if not lines:
        sys.exit(f"剧本 {args.script} 中没有 lines")

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # 读旧 manifest 用于断点续传（保留之前的 audio_length_ms 等元数据）
    manifest_path = os.path.join(out_dir, "manifest.json")
    old_entries = {}
    if os.path.isfile(manifest_path) and not args.force:
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                old = json.load(f)
            for entry in old.get("lines", []):
                old_entries[entry["idx"]] = entry
        except (json.JSONDecodeError, KeyError):
            old_entries = {}

    new_entries = []
    ok = fail = skip = 0
    total_ms = 0
    print(f"[synth] 开始合成 {len(lines)} 句 → {out_dir}/", file=sys.stderr)

    for i, line in enumerate(lines):
        idx = line.get("idx", i)
        role_name = line.get("role", "旁白")
        role = roles.get(role_name, {"voice_id": "audiobook_female_1"})
        merged = _merge_line_with_role(line, role)
        out_path = os.path.join(out_dir, f"line_{idx:03d}.{args.format}")

        # 续传：文件存在 + 旧条目无 error 才跳；之前失败的会重试
        if (not args.force
                and os.path.isfile(out_path)
                and os.path.getsize(out_path) > 0
                and idx in old_entries
                and not old_entries[idx].get("error")
                and old_entries[idx].get("file") == os.path.basename(out_path)):
            new_entries.append(old_entries[idx])
            skip += 1
            total_ms += int(old_entries[idx].get("audio_length_ms", 0))
            print(f"  [{idx:03d}] skip  ({role_name}: {merged['text'][:24]}…)", file=sys.stderr)
            continue

        print(f"  [{idx:03d}] {role_name:8s} | {merged['text'][:30]}…", file=sys.stderr)
        try:
            res = _post_tts(
                text=merged["text"],
                voice_id=merged["voice_id"],
                emotion=merged["emotion"],
                vol=merged["vol"],
                speed=merged["speed"],
                pitch=merged["pitch"],
                pronunciation=merged["pronunciation"],
                tts_model=script.get("tts_model", DEFAULT_TTS_MODEL),
                lang_boost=script.get("language_boost", DEFAULT_LANG_BOOST),
                audio_format=args.format,
            )
        except Exception as e:
            print(f"  [{idx:03d}] FAIL: {e}", file=sys.stderr)
            new_entries.append({
                "idx": idx, "role": role_name, "voice_id": merged["voice_id"],
                "file": None, "audio_length_ms": 0, "sample_rate": 0,
                "channels": 0, "format": args.format,
                "pause_after_ms": merged["pause_after_ms"], "error": str(e),
            })
            fail += 1
            continue

        with open(out_path, "wb") as f:
            f.write(res["audio_bytes"])

        entry = {
            "idx":            idx,
            "role":           role_name,
            "voice_id":       merged["voice_id"],
            "emotion":        merged["emotion"],
            "vol":            merged["vol"],
            "speed":          merged["speed"],
            "pitch":          merged["pitch"],
            "text":           merged["text"],
            "file":           os.path.basename(out_path),
            "audio_length_ms":res["audio_length_ms"],
            "sample_rate":    res["sample_rate"],
            "channels":       res["channels"],
            "format":         res["format"],
            "usage_chars":    res["usage_chars"],
            "pause_after_ms": merged["pause_after_ms"],
        }
        new_entries.append(entry)
        total_ms += res["audio_length_ms"]
        ok += 1

    # 按 idx 排序后落盘
    new_entries.sort(key=lambda e: e["idx"])
    manifest = {
        "title":         script.get("title", ""),
        "tts_model":     script.get("tts_model", DEFAULT_TTS_MODEL),
        "language_boost":script.get("language_boost", DEFAULT_LANG_BOOST),
        "total_audio_ms":total_ms,
        "lines":         new_entries,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[synth] 完成：成功 {ok} / 跳过 {skip} / 失败 {fail}，"
          f"总音频 {total_ms/1000:.2f}s，manifest → {manifest_path}")


# ===================== check 子命令 =====================

# 对话 attribution 动词（后接：「" 或 ：" 形式）
ATTRIBUTION_VERBS = "[说道叫道笑道喊道答道喝道陪笑道冷道接口道]"

# 对白里几乎不该出现的强叙述信号（出现即视为 narration 混入对白）
NARRATION_MARKERS = (
    "这么奉承",   # e.g. "……这么奉承一番。"
    "说的是",     # e.g. "……说的是北方口音。"
    "便道",       # 旁白用的转述动词
    "暗自",       # 心理叙述
    "不料",       # 旁白用转折
    "却见",       # 旁白用场景切换
)


def _normalize_for_coverage(s):
    """去掉所有引号、空白、轻微标点，用于跨格式覆盖率比对。"""
    # 用 \u 转义，避免源码里 ASCII " 与中文 “” 混淆
    return re.sub(
        r'[\u0022\u0027\u201c\u201d\u2018\u2019'
        r'\u300c\u300d\u3001\uff0c\uff1a\uff1b'
        r'\u2026\u2014\s]',
        '', s,
    )


def _split_sentences_keep_delim(text):
    """按 。！？；\n 拆句，保留分隔符。"""
    parts = re.split(r"([。！？；\n]+)", text)
    out, buf = [], ""
    for p in parts:
        if re.fullmatch(r"[。！？；\n]+", p or ""):
            buf += p
            if buf.strip():
                out.append(buf)
            buf = ""
        else:
            buf += p
    if buf.strip():
        out.append(buf)
    return out


def _longest_substr(haystack_norm, needle_norm, max_len=40):
    """needle 在 haystack 中最长覆盖子串长度。"""
    n = len(needle_norm)
    if n == 0:
        return 0
    best = 0
    upper = min(n, max_len)
    for L in range(upper, 2, -1):
        for i in range(0, n - L + 1):
            if needle_norm[i:i + L] in haystack_norm:
                return L
    return best


def cmd_check(args):
    """检查 script.json 与原 txt 的完整性，列出问题。"""
    if not os.path.isfile(args.input):
        sys.exit(f"找不到原文件: {args.input}")
    if not os.path.isfile(args.script):
        sys.exit(f"找不到剧本: {args.script}")

    original = open(args.input, encoding="utf-8").read()
    script = json.load(open(args.script, encoding="utf-8"))
    lines = script.get("lines", [])
    json_concat = "".join(line.get("text", "") for line in lines)
    json_norm = _normalize_for_coverage(json_concat)

    issues = []

    # --- Check 1: 短句覆盖率 ---
    for sent in _split_sentences_keep_delim(original):
        sn = _normalize_for_coverage(sent)
        if len(sn) < 4:
            continue
        if sn in json_norm:
            continue
        # 部分覆盖：找出 needle 在 haystack 中最长公共子串
        L = _longest_substr(json_norm, sn, max_len=min(len(sn), 30))
        coverage = L / len(sn) if sn else 0
        sev = "error" if coverage < 0.5 else "warn"
        preview = sent.strip().replace("\n", " ")
        if len(preview) > 60:
            preview = preview[:60] + "…"
        issues.append({
            "type":     "low_coverage",
            "severity": sev,
            "coverage": f"{coverage * 100:.0f}%",
            "detail":   f"短句覆盖 {coverage * 100:.0f}%；原文：{preview}",
        })

    # --- Check 2: attribution 完整性（旁白→对话 边界）---
    by_idx = {line.get("idx", i): line for i, line in enumerate(lines)}
    for line in lines:
        if line.get("role") != "旁白":
            continue
        idx = line.get("idx", -1)
        next_line = by_idx.get(idx + 1)
        if not next_line or next_line.get("role") == "旁白":
            continue
        t = line.get("text", "")
        # 正常结尾：含 attribution + 开引号
        if re.search(ATTRIBUTION_VERBS + r'[：:]"?$', t):
            continue
        # 旁白后是 dialogue，但旁白行末没有 attribution → 可能丢了归属
        issues.append({
            "type":     "attribution_likely_lost",
            "severity": "error",
            "idx":      idx,
            "detail":   f"旁白 [{idx}]「{t[-15:]}」后紧跟 {next_line.get('role')} "
                        f"[{next_line.get('idx')}]，但缺归属动词（道/笑道/叫道 等），"
                        f"可能丢失 attribution",
        })

    # --- Check 3: 对白混入叙述词 ---
    for line in lines:
        if line.get("role") == "旁白":
            continue
        t = line.get("text", "")
        for marker in NARRATION_MARKERS:
            if marker in t:
                preview = t.replace("\n", " ")
                if len(preview) > 50:
                    preview = preview[:50] + "…"
                issues.append({
                    "type":     "narration_in_dialogue",
                    "severity": "error",
                    "idx":      line.get("idx"),
                    "role":     line.get("role"),
                    "detail":   f"{line.get('role')} [{line.get('idx')}] 对白含叙述词「{marker}」：{preview}",
                })
                break  # 每行只报一个最强的 marker

    # --- Check 4: 旁白末尾残留 attribution 动词（疑似丢开引号）---
    for line in lines:
        if line.get("role") != "旁白":
            continue
        t = line.get("text", "")
        # 如：旁白行末尾是 `...道：` （没有后续 "）→ 警告
        if re.search(ATTRIBUTION_VERBS + r"[：:]$", t):
            issues.append({
                "type":     "missing_opening_quote",
                "severity": "warn",
                "idx":      line.get("idx"),
                "detail":   f"旁白 [{line.get('idx')}]「{t[-10:]}」以「：」结尾但缺开引号 \"",
            })

    # --- 报告 ---
    errors = [i for i in issues if i["severity"] == "error"]
    warns  = [i for i in issues if i["severity"] == "warn"]

    if not issues:
        print("✓ 检查通过，未发现问题")
        return 0

    print(f"共 {len(issues)} 个问题（{len(errors)} 错 / {len(warns)} 警）：\n")
    for i, iss in enumerate(issues, 1):
        sev = "❌" if iss["severity"] == "error" else "⚠️ "
        print(f"  {i:3d}. {sev} [{iss['type']}] {iss['detail']}")

    failed = bool(errors) or (args.strict and bool(warns))
    return 1 if failed else 0


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

    p_sy = sub.add_parser("synth", help="读 script.json，逐句调 MiniMax TTS 产出音频片段")
    p_sy.add_argument("--script",  "-s", required=True, help="script.json（analyze 产出）")
    p_sy.add_argument("--out-dir", "-o", default="output", help="输出目录（默认 output）")
    p_sy.add_argument("--format",   default="mp3", choices=["mp3", "pcm", "flac"],
                      help="音频格式（默认 mp3；如要 wav 直接合并请用 pcm）")
    p_sy.add_argument("--force",    action="store_true", help="忽略已存在文件，重新合成全部")
    p_sy.set_defaults(func=cmd_synth)

    p_ck = sub.add_parser("check", help="检查 script.json 与原始 txt 的完整性（覆盖率、attribution 丢失、对白混入叙述等）")
    p_ck.add_argument("--input",  "-i", required=True, help="原始小说 .txt")
    p_ck.add_argument("--script", "-s", required=True, help="待检查的 script.json")
    p_ck.add_argument("--strict", action="store_true",
                      help="把 warn 也算作 error，非零退出码")
    p_ck.set_defaults(func=cmd_check)

    # 后续子命令占位（merge / all）暂未实现
    for name in ("merge", "all"):
        p = sub.add_parser(name, help="[未实现] 后续阶段")
        p.set_defaults(func=lambda a, _n=name: sys.exit(f"[{_n}] 尚未实现"))

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()