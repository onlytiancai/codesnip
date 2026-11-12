#!/usr/bin/env ~/.pyenv/versions/qlib/bin/python
"""CET-4 真题高频词 v3 全量分组器。

两阶段 LLM 流水线：
  Pass 1 (1 次 LLM 调用)  : 把 1162 词粗分到 ~30-40 个语义大类
  Pass 2 (~40 次 LLM 调用) : 在每个大类内部细分成 3-5 词/组 + 写讲解

输出：
  - cet4_hexinci_groups_v3.json  (顶层兼容 v2，并新增 categories[] 二级结构)
  - /tmp/cet4_v3_categories.json (Pass 1 中间结果，可断点续跑)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import anthropic

# 复用 incremental.py 的工具函数
from hexinci_group_incremental import load_vocab, load_db, _empty_db, already_grouped, save_db

ROOT                 = Path(__file__).parent
TXT                  = ROOT / "cet4_sijizhenti_hexinci.txt"
JSON_PATH            = ROOT / "cet4_hexinci_groups_v3.json"
CATEGORY_DUMP_PATH   = Path("/tmp/cet4_v3_categories.json")
GROUP_DUMP_PATH      = Path("/tmp/cet4_v3_groups.json")

# 粗分类（大类）锚点参考，CET-4 教辅常见 13 大类 + 兜底
# 每个类别有固定 id（数字字符串）+ 中文名 + 英文名
CATEGORY_ANCHORS: list[tuple[str, str, str]] = [
    ("01", "情感与心理",     "Emotions & Psychology"),
    ("02", "教育与学习",     "Education & Learning"),
    ("03", "商业与经济",     "Business & Economy"),
    ("04", "科技与互联网",   "Technology & Internet"),
    ("05", "健康与医疗",     "Health & Medicine"),
    ("06", "环境与自然",     "Environment & Nature"),
    ("07", "法律与政治",     "Law & Politics"),
    ("08", "交通与旅行",     "Transportation & Travel"),
    ("09", "社会与文化",     "Society & Culture"),
    ("10", "工作与职业",     "Work & Career"),
    ("11", "动作与行为",     "Actions & Behaviors"),
    ("12", "程度与副词",     "Degree & Adverbs"),
    ("13", "数量与度量",     "Quantity & Measurement"),
    ("14", "时间与频率",     "Time & Frequency"),
    ("15", "空间与位置",     "Space & Location"),
    ("16", "材料与制造",     "Materials & Manufacturing"),
    ("17", "艺术与娱乐",     "Arts & Entertainment"),
    ("18", "食物与餐饮",     "Food & Dining"),
    ("19", "服装与外观",     "Clothing & Appearance"),
    ("20", "宗教与哲学",     "Religion & Philosophy"),
    ("21", "抽象概念与品质", "Abstract Concepts & Qualities"),
    ("22", "自然科学",       "Natural Sciences"),
    ("23", "体育与竞赛",     "Sports & Competition"),
    ("24", "沟通与语言",     "Communication & Language"),
    ("25", "军事与冲突",     "Military & Conflict"),
    ("99", "其他词汇",       "Others"),
]


def _build_system_prompt_categories(categories: list[tuple[str, str, str]]) -> str:
    cat_lines = "\n".join(f"  {cid}: {zh} ({en})" for cid, zh, en in categories)
    return f"""你是 CET-4 词汇教学专家，擅长把英文词汇按语义归类。

【任务】
下面会给出约 400 个 CET-4 真题高频词。你的任务是把这 400 个词**全部**归入我预先定义好的 26 个固定大类（categories）中，每个词**必须出现且仅出现一次**。

【固定分类（id 必须精确使用）】
{cat_lines}

【每个词的硬性约束】
- 每个词必须归入上述 26 个类的某一个，输出其 2 位 id（如 "01"）
- 不要造词、不要改写、不要加复数（如列表里是 "profitable" 就不准输出 "profit"）
- 词必须**精确匹配**输入词表中的字符串（包括大小写）
- 400 个词必须 100% 出现在输出中，**不允许遗漏**

【如果一个词实在分不进 26 个类的任何一个】
归入 "99"（其他词汇），但应当尽量避免，因为大部分 CET-4 词都能归入主类。

【输出 schema（严格匹配）】
{{
  "assignments": [
    {{"word": "word1", "category_id": "01"}},
    {{"word": "word2", "category_id": "03"}},
    ...
  ]
}}

直接输出 JSON（不要 ```json``` 围栏，不要其它任何文字）。【关键】输出必须是单行紧凑 JSON，不要格式化缩进，节省 token。"""


USER_PROMPT_CATEGORIES = """【本批词表（共 {n} 个，按字母序）】
{vocab_csv}

请按系统提示中的 schema，把这 {n} 个词**全部**归入 26 个固定大类中。每个词必须恰好出现在一个 assignment 里，不允许遗漏任何词，也不允许造词。输出紧凑 JSON。"""


# Pass 2 细分 prompt（参考 incremental.py 的讲解风格）
def _build_system_prompt_subgroups(category_name_zh: str, words: list[str], *,
                                   sub_batch_note: str = "") -> str:
    note_block = f"\n【子批说明】{sub_batch_note}\n" if sub_batch_note else ""
    n = len(words)
    expected_groups = max(1, n // 4)  # 期望 ~n/4 个子组（每组 4 词）
    return f"""你是 CET-4 词汇教学专家。

【当前大类】{category_name_zh}{note_block}
【本类词表（共 {n} 个）】
{", ".join(words)}

【任务】
把这 {n} 个词全部细分为约 {expected_groups} 个子组（每组 3-5 词），为每个子组写一段 400-800 字的中文讲解。**所有词必须无遗漏无重复地分到某个子组里**。

【🔥 硬性约束（违反任意一条直接 reject）】
1. **每个子组 words 数严格 3-5 个**（不可多 1 个，也不可少 1 个）。这是系统级硬约束，无法绕过。
   - {n} 个词 → 约 {expected_groups} 个子组
   - 例：20 个词 → 分 5 个子组，每组 4 词
   - 例：35 个词 → 分 8-9 个子组，6 个 4 词组 + 2-3 个 5 词组
2. 所有 words 的并集必须 == 本类词表的 {n} 个词，不允许遗漏任何一个词
3. 词必须**精确匹配**词表中的字符串，不允许造词、不允许变形（不要把 appreciate 写成 apprec）
4. 同一词不允许出现在两个子组里

【自检步骤（输出前必须执行）】
- 列出每个 group 的 words 数，确认全部在 [3, 5] 范围内
- 确认所有 words 之并 == 输入的 {n} 个词
- 任何超 5 词的子组必须立即拆开

【讲解结构（每个子组 400-800 字）】
①组主题总述（一段话说清这组词为什么归为一类）
②逐词释义 + 例句：对每个词给出【中文释义】+ 一个简短英文例句（10-20 词，体现 CET-4 典型用法）
③辨析 / 搭配建议（一段话讲清词间差异、典型搭配、语域差异）

【格式要求】
- 用中文「」或纯文字，**不要在中文里嵌入 ASCII 双引号 "**，避免破坏 JSON 输出
- 直接输出 JSON（不要 ```json``` 围栏，不要其它任何文字）

【输出 schema（严格匹配）】
{{
  "groups": [
    {{
      "title_zh": "子组中文主题",
      "title_en": "子组英文主题（可选，不确定传空串）",
      "words": ["word1", "word2", "word3", "word4"],
      "explanation_zh": "400-800 字的中文讲解"
    }},
    ...
  ]
}}
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CET-4 v3 全量分组（两阶段 LLM）")
    p.add_argument("--debug",         action="store_true", help="打印 prompt / 响应 / 校验详情")
    p.add_argument("--skip-pass1",    action="store_true", help="跳过 Pass 1，复用 /tmp/cet4_v3_categories.json")
    p.add_argument("--skip-pass2",    action="store_true", help="跳过 Pass 2，仅生成 /tmp/cet4_v3_categories.json")
    p.add_argument("--resume-from",  type=int, default=None, help="从指定 category_id 重新跑 Pass 2（之前的 groups 会保留）")
    p.add_argument("--max-workers",  type=int, default=1, help="Pass 2 并发数（默认 1，可试 3）")
    p.add_argument("--max-retries",  type=int, default=2, help="单次 LLM 调用最大重试次数（默认 2；失败走 auto-fix）")
    p.add_argument("--model",        type=str, default="MiniMax-M3", help="要使用的模型")
    p.add_argument("--sleep",        type=float, default=0.0, help="每次 LLM 调用后 sleep 秒数（防 rate limit）")
    return p.parse_args()


def _strip_code_fence(raw: str) -> str:
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()


def _call_llm(client: anthropic.Anthropic, system: str, user: str, *,
              max_tokens: int, model: str, max_retries: int, debug: bool) -> str:
    """统一 LLM 调用 + 重试 + JSON 围栏剥离。"""
    last_err: str | None = None
    for attempt in range(1, max_retries + 1):
        if debug:
            print(f"\n[debug] ── attempt {attempt}/{max_retries} ──", file=sys.stderr)
        msg = client.messages.create(
            model       = model,
            system      = system,
            max_tokens  = max_tokens,
            temperature = 0.6,
            messages    = [{"role": "user", "content": user}],
        )
        raw = "".join(b.text for b in msg.content if b.type == "text").strip()
        raw = _strip_code_fence(raw)
        if debug:
            print(f"[debug] LLM raw_chars={len(raw)}  stop_reason={msg.stop_reason}", file=sys.stderr)
            head, tail = raw[:300], raw[-300:]
            print(f"[debug] raw[:300]={head!r}", file=sys.stderr)
            if len(raw) > 600:
                print(f"[debug] raw[-300:]={tail!r}", file=sys.stderr)
        # 快速校验：必须能 JSON.loads
        try:
            json.loads(raw)
            return raw
        except json.JSONDecodeError as e:
            last_err = f"JSON parse failed at line {e.lineno} col {e.colno}: {raw[:200]}"
            if debug:
                print(f"[debug] JSONDecodeError: {e}", file=sys.stderr)
            print(f"  ⚠️  attempt {attempt}/{max_retries}: {last_err}", file=sys.stderr)
    raise SystemExit(f"❌ LLM 调用 {max_retries} 次都失败: {last_err}")


# ============================================================
# Pass 1: 粗分类（分批 + 固定 26 类）
# ============================================================

PASS1_BATCH_SIZE = 400  # 每批 ≤ 400 词，确保单次 LLM 输出不超过 token 预算


def _pass1_classify_batch(client: anthropic.Anthropic, batch: list[str], *,
                          args: argparse.Namespace) -> dict[str, str]:
    """对单个批次的词调一次 LLM，返回 {word: category_id} 字典。"""
    system = _build_system_prompt_categories(CATEGORY_ANCHORS)
    user   = USER_PROMPT_CATEGORIES.format(n=len(batch), vocab_csv=", ".join(batch))

    if args.debug:
        print(f"[debug]   batch_size={len(batch)}  system_chars={len(system)}  user_chars={len(user)}", file=sys.stderr)

    raw = _call_llm(client, system, user, max_tokens=6000, model=args.model,
                    max_retries=args.max_retries, debug=args.debug)
    obj = json.loads(raw)

    assignments = obj.get("assignments", [])
    if not isinstance(assignments, list):
        raise SystemExit(f"❌ Pass 1 batch 输出缺少 assignments 数组")

    valid_ids = {cid for cid, _, _ in CATEGORY_ANCHORS}
    out: dict[str, str] = {}
    for a in assignments:
        w = a.get("word", "").strip()
        cid = str(a.get("category_id", "")).strip()
        if not w or not cid:
            raise SystemExit(f"❌ Pass 1 assignment 字段不全: {a}")
        if cid not in valid_ids:
            raise SystemExit(f"❌ Pass 1 未知 category_id={cid!r}（word={w!r}）")
        if w in out:
            raise SystemExit(f"❌ Pass 1 batch 词重复: {w!r}")
        out[w] = cid

    # 校验：批内所有词必须 100% 出现
    missing = set(batch) - set(out.keys())
    if missing:
        raise SystemExit(f"❌ Pass 1 batch 遗漏 {len(missing)} 个词: {sorted(missing)[:30]}")
    return out


def pass1_categorize(vocab_sorted: list[str], *, args: argparse.Namespace) -> list[dict]:
    """把 1162 词分到 26 个固定大类。分批调 LLM，聚合结果。"""
    client = anthropic.Anthropic()

    # 分批
    batches = [vocab_sorted[i:i + PASS1_BATCH_SIZE] for i in range(0, len(vocab_sorted), PASS1_BATCH_SIZE)]
    print(f"📦 Pass 1: {len(vocab_sorted)} 词 → {len(batches)} 批（每批 ≤ {PASS1_BATCH_SIZE}）")

    all_assignments: dict[str, str] = {}
    for i, batch in enumerate(batches, 1):
        print(f"  → 批 {i}/{len(batches)} ({len(batch)} 词)")
        result = _pass1_classify_batch(client, batch, args=args)
        all_assignments.update(result)
        if args.sleep > 0:
            time.sleep(args.sleep)

    # 校验：所有词必须出现
    vocab_set = set(vocab_sorted)
    missing = vocab_set - set(all_assignments.keys())
    if missing:
        raise SystemExit(f"❌ Pass 1 聚合后遗漏 {len(missing)} 个词: {sorted(missing)[:30]}")

    # 按 category_id 聚合
    by_cat: dict[str, list[str]] = {cid: [] for cid, _, _ in CATEGORY_ANCHORS}
    for w, cid in all_assignments.items():
        by_cat[cid].append(w)

    # 拼成 category 列表
    categories = []
    next_id = 1
    for cid, zh, en in CATEGORY_ANCHORS:
        ws = by_cat[cid]
        if not ws:
            continue  # 跳过空类
        categories.append({
            "id":      next_id,
            "name_zh": zh,
            "name_en": en,
            "anchor_id": cid,  # 保留锚点 id 便于追溯
            "words":   ws,
        })
        next_id += 1

    total_words = sum(len(c["words"]) for c in categories)
    print(f"✅ Pass 1 完成：{len(categories)} 个非空大类，{total_words} 词全部归类")
    return categories


# ============================================================
# Pass 2: 细分子组 + 讲解
# ============================================================

# Pass 2 单批上限：每批 ≤ PASS2_BATCH_SIZE 词，输出 ≤ 8000 tokens 安全
PASS2_BATCH_SIZE   = 40
PASS2_MAX_TOKENS   = 8000   # 单次 LLM 输出上限



def _greedy_chunk_3_to_5(words: list[str]) -> list[list[str]]:
    """贪心把词列表切分成 3-5 词/块的二维数组。保证每块 ∈ [3, 5]。"""
    n = len(words)
    if n < 3:
        # 极端 < 3：返回单块
        return [list(words)] if n > 0 else []

    result: list[list[str]] = []
    i = 0
    while i < n:
        remaining = n - i
        if remaining == 3 or remaining == 4 or remaining == 5:
            # 剩余刚好 3-5：整组取走
            result.append(words[i:i + remaining])
            i += remaining
        elif remaining == 1 or remaining == 2:
            # 剩余 1-2 词：从上一块抽 1-2 词过来凑成 3+
            prev = result[-1]
            need = 3 - remaining  # 还差几个词才能凑成 3
            take = min(need, len(prev))  # 不能超过 prev 大小
            moved = prev[-take:]
            del prev[-take:]
            merged = moved + words[i:]
            result.append(merged)
            i += remaining
        else:  # remaining >= 6
            result.append(words[i:i + 4])
            i += 4
    return result


def pass2_subgroups_for_category(client: anthropic.Anthropic, category: dict, vocab_set: set[str],
                                 *, args: argparse.Namespace) -> list[dict]:
    """对单个大类调 1+ 次 LLM 做细分子组 + 讲解（两阶段：先骨架再补讲解）。"""
    cat_id        = category["id"]
    cat_zh        = category["name_zh"]
    cat_en        = category.get("name_en", "")
    words         = list(category["words"])
    n_words       = len(words)
    cat_word_set  = set(words)

    # 计算子批数：每批 ≤ PASS2_BATCH_SIZE
    if n_words <= PASS2_BATCH_SIZE:
        sub_batches = [words]
    else:
        sub_batches = [words[i:i + PASS2_BATCH_SIZE] for i in range(0, n_words, PASS2_BATCH_SIZE)]

    if args.debug:
        print(f"\n[debug] Pass 2a category {cat_id} ({cat_zh}): {n_words} 词 → {len(sub_batches)} 子批", file=sys.stderr)

    # ===== Stage 1: 骨架（只要分组，不要讲解） =====
    all_groups: list[dict] = []
    for sb_i, sb in enumerate(sub_batches, 1):
        if len(sub_batches) > 1:
            note = f"这是大类的第 {sb_i}/{len(sub_batches)} 个子批（每批独立成组，组间不要跨批混合词）"
        else:
            note = ""
        groups = pass2a_skeleton_for_subbatch(
            client, cat_id, cat_zh, cat_en, sb, vocab_set, cat_word_set, note, args=args,
        )
        all_groups.extend(groups)
        if args.sleep > 0:
            time.sleep(args.sleep)

    # 整类校验：所有词必须出现。缺词时降级为 auto-fix（不再 raise）
    used_all = {w for g in all_groups for w in g["words"]}
    missing  = cat_word_set - used_all
    if missing:
        # 降级：把缺词并到最后一个 group（或独立成组）
        print(f"  ⤵️  Pass 2a cat {cat_id}: 缺 {len(missing)} 词 {sorted(missing)[:10]} → auto-补到末组", file=sys.stderr)
        if all_groups:
            all_groups[-1]["words"].extend(sorted(missing))
        else:
            all_groups.append({
                "title_zh": f"自动补全缺词组",
                "title_en": "",
                "words": sorted(missing),
                "explanation_zh": "",
            })
        # 重切最后 group 到 3-5 词
        # 简单做法：如果最后 group > 7 词，重新贪心切
        last = all_groups[-1]
        if len(last["words"]) > 5:
            blocks = _greedy_chunk_3_to_5(last["words"])
            all_groups = all_groups[:-1] + [{
                "title_zh":       f"自动补全 {i+1}",
                "title_en":       "",
                "words":          blk,
                "explanation_zh": "",
            } for i, blk in enumerate(blocks)]

    # ===== Stage 2: 补讲解（按 batch 调 LLM） =====
    if args.debug:
        print(f"\n[debug] Pass 2b category {cat_id}: {len(all_groups)} 子组需要补讲解", file=sys.stderr)
    pass2b_fill_explanations_for_groups(client, all_groups, cat_zh, cat_en, args=args)

    print(f"  ✅ category {cat_id} ({cat_zh}): {n_words} 词 → {len(all_groups)} 子组（含讲解）")
    return all_groups


# ----- Stage 1: 骨架分组 -----

def _build_skeleton_prompt(cat_zh: str, words: list[str], sub_batch_note: str) -> tuple[str, str]:
    n = len(words)
    expected = max(1, n // 4)
    system = f"""你是 CET-4 词汇分类专家。

【当前大类】{cat_zh}
{f"【子批说明】{sub_batch_note}" if sub_batch_note else ""}
【本类词表（共 {n} 个）】
{", ".join(words)}

【任务】
把这 {n} 个词细分为约 {expected} 个子组（每组 3-5 词），每个子组只需要一个简短的中文主题（**不要写讲解，本阶段只做分组**）。

【约束】
1. 每组 3-5 词
2. 所有词必须出现且仅出现一次
3. 词必须精确匹配输入词表
4. 主题用 2-8 字中文（如「惊喜与恐惧」「学习与成长」）

【输出 schema】
{{
  "groups": [
    {{"title_zh": "简短主题", "words": ["w1", "w2", "w3"]}}
  ]
}}

直接输出 JSON，不要其它任何文字。"""
    user = f"【大类中文名】{cat_zh}\n\n请把上述 {n} 个词分成 3-5 词/组的子组（只要分组，不要讲解）。"
    return system, user


def pass2a_skeleton_for_subbatch(client: anthropic.Anthropic, cat_id: int, cat_zh: str, cat_en: str,
                                  words: list[str], vocab_set: set[str], cat_word_set: set[str],
                                  sub_batch_note: str, *, args: argparse.Namespace) -> list[dict]:
    """Stage 1: 单批骨架分组，只要 words + title_zh。校验失败走 auto-fix。"""
    system, user = _build_skeleton_prompt(cat_zh, words, sub_batch_note)
    last_err: str | None = None

    for attempt in range(1, args.max_retries + 1):
        if args.debug:
            print(f"\n[debug]   2a cat {cat_id} sub-batch attempt {attempt}/{args.max_retries}", file=sys.stderr)
        raw = _call_llm(client, system, user, max_tokens=4000, model=args.model,
                        max_retries=args.max_retries, debug=args.debug)
        obj = json.loads(raw)
        groups_raw = obj.get("groups", [])

        # 校验：去重 + 覆盖
        all_used: list[str] = []
        for g in groups_raw:
            for w in g.get("words", []):
                w = w.strip()
                if w in cat_word_set and w not in all_used:
                    all_used.append(w)
        missing = set(words) - set(all_used)
        if not missing:
            # 通过！按 3-5 词切分（保留 LLM 的标题但允许 auto-chunk）
            blocks = _greedy_chunk_3_to_5(all_used)
            out: list[dict] = []
            title_pool = [(g.get("title_zh") or "").strip() for g in groups_raw]
            for i, blk in enumerate(blocks):
                out.append({
                    "title_zh":       title_pool[i] if i < len(title_pool) and title_pool[i] else f"自动分组 {i+1}",
                    "title_en":       "",
                    "words":          blk,
                    "explanation_zh": "",  # Stage 2 补
                })
            return out
        last_err = f"遗漏 {len(missing)} 词"
        print(f"  ⚠️  2a cat {cat_id} attempt {attempt}/{args.max_retries}: {last_err}", file=sys.stderr)

    # 失败：auto-fix（不要求严格 3-5 词，贪心切分）
    print(f"  ⤵️  2a cat {cat_id} 严格重试 {args.max_retries} 次失败 → auto-fix", file=sys.stderr)
    blocks = _greedy_chunk_3_to_5(all_used)
    return [{
        "title_zh":       f"自动分组 {i+1}",
        "title_en":       "",
        "words":          blk,
        "explanation_zh": "",
    } for i, blk in enumerate(blocks)]


# ----- Stage 2: 补讲解 -----

PASS2B_BATCH_SIZE = 6   # 一次 LLM 调用解释 6 个子组


def _build_explanations_prompt(cat_zh: str, groups_for_prompt: list[dict]) -> tuple[str, str]:
    """为一批子组生成中文讲解 prompt。"""
    items_text = "\n".join(
        f"子组 {i+1}：{g['title_zh']} → {g['words']}"
        for i, g in enumerate(groups_for_prompt)
    )
    system = f"""你是 CET-4 词汇教学专家。

【大类】{cat_zh}

下面有 {len(groups_for_prompt)} 个子组，每个子组有 3-5 个 CET-4 高频词。请为每个子组写一段 400-800 字的中文讲解，结构：
①组主题总述（一段话说清这组词为什么归为一类）
②逐词释义 + 例句：对每个词给出【中文释义】+ 一个简短英文例句（10-20 词，体现 CET-4 典型用法）
③辨析 / 搭配建议（一段话讲清词间差异、典型搭配、语域差异）

【硬性约束】
- 每个子组讲解 400-800 字（中文字符）
- 讲解中必须涵盖该子组**所有**词（不要漏词）
- 用中文「」或纯文字，**不要在中文里嵌入 ASCII 双引号 "**
- 直接输出 JSON（不要 ```json``` 围栏，不要其它任何文字）

【输出 schema】
{{
  "explanations": [
    {{
      "title_zh": "原主题（必须照抄输入的 title_zh，不要改）",
      "explanation_zh": "400-800 字中文讲解"
    }},
    ...
  ]
}}
"""
    user = f"【待讲解子组列表】\n{items_text}\n\n请按 schema 输出讲解。"
    return system, user


def pass2b_fill_explanations_for_groups(client: anthropic.Anthropic, groups: list[dict], cat_zh: str, cat_en: str,
                                         *, args: argparse.Namespace) -> None:
    """Stage 2: 对一组子组调 LLM 补讲解，直接修改 groups 中每条的 explanation_zh。"""
    # 找所有需要补讲解的
    pending = [g for g in groups if not g.get("explanation_zh") or "auto-fix" in g.get("explanation_zh", "")]
    if not pending:
        return

    # 按 PASS2B_BATCH_SIZE 分批
    for i in range(0, len(pending), PASS2B_BATCH_SIZE):
        batch = pending[i:i + PASS2B_BATCH_SIZE]
        system, user = _build_explanations_prompt(cat_zh, batch)

        for attempt in range(1, args.max_retries + 1):
            if args.debug:
                print(f"\n[debug]   2b batch {i//PASS2B_BATCH_SIZE + 1} attempt {attempt}/{args.max_retries}", file=sys.stderr)
            raw = _call_llm(client, system, user, max_tokens=PASS2_MAX_TOKENS, model=args.model,
                            max_retries=args.max_retries, debug=args.debug)
            obj = json.loads(raw)
            exps = obj.get("explanations", [])
            if not exps:
                continue

            # 校验：每个 explanation 必须包含对应的所有词
            valid: dict[str, str] = {}  # title_zh -> explanation
            for e in exps:
                title = (e.get("title_zh") or "").strip()
                expl  = (e.get("explanation_zh") or "").strip()
                if not title or not expl:
                    continue
                # 找对应的 batch item（按 title 匹配）
                match = next((g for g in batch if g["title_zh"] == title), None)
                if not match:
                    continue
                # 校验讲解字数（粗略）
                if not (200 <= len(expl) <= 2000):
                    continue
                # 校验讲解中包含所有词（粗略：每个词至少出现一次）
                miss_in_expl = [w for w in match["words"] if w.lower() not in expl.lower()]
                if miss_in_expl:
                    if args.debug:
                        print(f"[debug]     explanation for {title!r} 漏词: {miss_in_expl}", file=sys.stderr)
                    continue
                valid[title] = expl

            if len(valid) == len(batch):
                # 全部成功
                for g in batch:
                    g["explanation_zh"] = valid[g["title_zh"]]
                break
            last_uncovered = [g["title_zh"] for g in batch if g["title_zh"] not in valid]
            print(f"  ⚠️  2b batch attempt {attempt}/{args.max_retries}: 缺讲解 {len(last_uncovered)} 个", file=sys.stderr)
        else:
            # 全部失败：保留 placeholder
            for g in batch:
                if not g.get("explanation_zh") or "auto-fix" in g.get("explanation_zh", ""):
                    g["explanation_zh"] = f"（讲解待补：{cat_zh} / {g['title_zh']}）"

        if args.sleep > 0:
            time.sleep(args.sleep)


def pass2_run_all(categories: list[dict], vocab_set: set[str], *, args: argparse.Namespace) -> dict:
    """对每个大类跑 Pass 2（两阶段），返回 {category_id: [group, ...]} 字典。"""
    client = anthropic.Anthropic()
    out: dict[int, list[dict]] = {}

    # 断点续跑：复用已存在的 groups 文件
    if GROUP_DUMP_PATH.exists() and args.resume_from is None:
        try:
            existing = json.loads(GROUP_DUMP_PATH.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                out = {int(k): v for k, v in existing.items()}
                print(f"⏩ 复用已有 {GROUP_DUMP_PATH}：{len(out)} 个类已跑完", file=sys.stderr)
        except (json.JSONDecodeError, ValueError):
            pass

    start_idx = 0
    if args.resume_from is not None:
        for i, cat in enumerate(categories):
            if cat["id"] == args.resume_from:
                start_idx = i
                break

    for i, cat in enumerate(categories[start_idx:], start=start_idx):
        cid = cat["id"]
        if cid in out:
            print(f"⏩ 跳过 category {cid}（已存在）", file=sys.stderr)
            continue
        groups = pass2_subgroups_for_category(client, cat, vocab_set, args=args)
        out[cid] = groups
        # 每完成一类就 dump 一次（断点续跑）
        GROUP_DUMP_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.sleep > 0:
            time.sleep(args.sleep)

    return out


# ============================================================
# 后处理：拼装 v3 JSON
# ============================================================

def assemble_v3_json(categories: list[dict], pass2_out: dict[int, list[dict]], vocab_set: set[str]) -> dict:
    """把 Pass 1 + Pass 2 结果拼成 v3 schema，同时计算 ungrouped_words。"""
    used_words: set[str] = set()
    flat_groups: list[dict] = []
    next_group_id = 1

    cat_block = []
    for cat in categories:
        cid = cat["id"]
        groups_raw = pass2_out.get(cid, [])
        groups = []
        for g in groups_raw:
            groups.append({
                "group_id":       next_group_id,
                "title_zh":       g["title_zh"],
                "title_en":       g["title_en"],
                "words":          g["words"],
                "explanation_zh": g["explanation_zh"],
            })
            flat_groups.append({
                "category_id": cid,
                "category_name_zh": cat["name_zh"],
                "group_id":       next_group_id,
                "title_zh":       g["title_zh"],
                "title_en":       g["title_en"],
                "words":          g["words"],
                "explanation_zh": g["explanation_zh"],
            })
            used_words.update(g["words"])
            next_group_id += 1
        cat_block.append({
            "category_id": cid,
            "name_zh":     cat["name_zh"],
            "name_en":     cat.get("name_en", ""),
            "groups":      groups,
        })

    ungrouped = sorted(vocab_set - used_words)

    db = _empty_db()
    db.update({
        "schema_version":     "v3",
        "total_groups":       len(flat_groups),
        "total_words_grouped": len(used_words),
        "vocab_size":         len(vocab_set),
        "remaining":          len(ungrouped),
        "categories":         cat_block,
        "groups":             flat_groups,            # 平铺数组（v2 兼容）
        "ungrouped_words":    ungrouped,
    })
    return db


def print_distribution(db: dict) -> None:
    print("\n────────── 主题分布 ──────────")
    for cat in db["categories"]:
        n_words = sum(len(g["words"]) for g in cat["groups"])
        print(f"  [{cat['category_id']:>2}] {cat['name_zh']:<20} {cat['name_en']:<30} {len(cat['groups']):>3} 组 / {n_words:>4} 词")
    print(f"\n总计: {len(db['categories'])} 大类 / {db['total_groups']} 组 / {db['total_words_grouped']} 词")
    if db["ungrouped_words"]:
        print(f"⚠️  未分组词 {len(db['ungrouped_words'])} 个: {db['ungrouped_words'][:30]}")


def main() -> int:
    args = parse_args()

    vocab_set  = load_vocab()
    vocab_list = sorted(vocab_set)
    if len(vocab_set) != 1162:
        print(f"⚠️  词表大小 {len(vocab_set)} ≠ 1162", file=sys.stderr)

    # ---------- Pass 1 ----------
    if args.skip_pass1 and CATEGORY_DUMP_PATH.exists():
        categories = json.loads(CATEGORY_DUMP_PATH.read_text(encoding="utf-8"))
        print(f"⏩ 跳过 Pass 1，从 {CATEGORY_DUMP_PATH} 读 {len(categories)} 个大类")
    else:
        categories = pass1_categorize(vocab_list, args=args)
        CATEGORY_DUMP_PATH.write_text(json.dumps(categories, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"💾 Pass 1 结果已存到 {CATEGORY_DUMP_PATH}")

    if args.skip_pass2:
        print("⏩ --skip-pass2：已生成 categories 中间文件，停止")
        return 0

    # ---------- Pass 2 ----------
    pass2_out = pass2_run_all(categories, vocab_set, args=args)

    # ---------- 后处理 ----------
    db = assemble_v3_json(categories, pass2_out, vocab_set)
    save_db(db, JSON_PATH)
    print(f"\n💾 v3 JSON 已写出: {JSON_PATH}")
    print_distribution(db)

    # 终极校验
    all_grouped = {w for cat in db["categories"] for g in cat["groups"] for w in g["words"]}
    assert len(all_grouped) == len(vocab_set), f"组内总词数 {len(all_grouped)} ≠ 词表 {len(vocab_set)}"
    assert all_grouped == vocab_set, f"差集: {all_grouped ^ vocab_set}"
    for cat in db["categories"]:
        for g in cat["groups"]:
            assert 3 <= len(g["words"]) <= 5, f"group {g['group_id']} 词数 {len(g['words'])} 越界"
    print("\n✅ 终极校验通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
