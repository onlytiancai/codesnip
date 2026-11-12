"""CLI: 用本地 ollama 模型给英文句子的某个后缀提供改写候选。

与 ``translator.py`` 不共享代码路径:那边走 Anthropic 的 ``tool_use``+大模型,
这边走本地 ollama(默认 ``s2021008840/hy-mt2:1.8b``),提示词必须用
中文指令+few-shot,不能用 ``translator.REPHRASE_SYSTEM`` 那套英文抽象规则,
1.8B 小模型跨长 prompt 维护约束的能力不够。

对外只暴露两件事:

- ``rephrase_candidates(source_zh, current, prefix, target_word, n=6,
  *, model, base_url) -> list[str]``
  返回完整英文句子(含 prefix)。
- ``main()`` 命令行。

异常表:
    ModelNotAvailableError -- ollama 起不来或模型没 pull。
    argparse 的 SystemExit(2) -- 缺参数。
    返回空 list 时脚本退码 1。

实现细节见同目录 ``/Users/huhao/.claude/plans/ollama-run-s2021008840-hy-mt2-1-8b-linear-kettle.md``。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import List, Optional

from ollama import ChatResponse, chat


# --- Configuration ---------------------------------------------------------


DEFAULT_MODEL = "s2021008840/hy-mt2:1.8b"


# Few-shot 例子必须能让小模型理解「每条以数字开头、用英文、句子完整」,
# 因此示例本身要尽量接近真实任务的形态:中文→英文改写一个动词。
_FEW_SHOT = """\
示例(把当前英文中的 "{ex_target}" 换成别的表达,前面 "{ex_prefix}" 保持不变):
中文:欢迎使用我们的产品
当前:Welcome {ex_prefix} {ex_current} our product.
1. Welcome {ex_prefix} try our product.
2. Welcome {ex_prefix} enjoy our product.
3. We're glad you picked our product.
4. Feel free to explore our product.
5. Thank you for choosing our product.
6. We hope you love our product."""


def _build_prompt(source_zh: str, current: str, prefix: str, target_word: str, n: int) -> str:
    """构造发给 ollama 的中文指令+few-shot(主路径:完整后缀)。

    关键设计:只让模型生成 *suffix*(target 位置之后的部分),
    代码侧负责 prepend prefix。1.8B 小模型跨长 prompt 维护约束的能力
    不足 —— 哪怕在 prompt 里写「prefix 必须保持原样」,它也会丢掉
    「feedback from using it」改写成更短的「Your usage insights will」之类。
    把"prefix 保留"放到代码里做确定性 prepend 比依赖模型稳得多。

    few-shot 设计要点(踩过的坑):
        - 不要用「verb + 固定宾语」这种可复用模式,否则小模型会照抄宾语到
          真实任务里(实测「verb + our product」会泄漏到无关句子)。
        - 6 行示例每行长度/结尾要不同,降低被「模板化」复制的概率。
        - 至少要有一个高频词(`is` 之类)的示例,让模型学到「即使是
          语法上必备的词也要换掉」。
    """
    rest = current
    pfx = prefix.strip()
    if pfx and rest.lower().startswith(pfx.lower()):
        rest = rest[len(pfx):].lstrip()
    few_shot = (
        "示例 A(把目标词换成别的表达,给出 {n} 个不同版本,每行是「{prefix}」之后的部分):\n"
        "中文:让我开始\n"
        "当前:Let me begin.\n"
        "待替换:begin\n"
        '"Let me" 之后的部分: begin.\n'
        "1. start now.\n"
        "2. go.\n"
        "3. proceed.\n"
        "4. move on to the next step.\n"
        "5. take action.\n"
        "6. get going right away.\n"
        "\n"
        "示例 B(目标词是高频 be 动词也要换,例如 is / was / are):\n"
        "中文:他在家\n"
        "当前:He is at home.\n"
        "待替换:is\n"
        '"He" 之后的部分: is at home.\n'
        "1. stays at home.\n"
        "2. remains at home.\n"
        "3. spends time at home.\n"
        "4. works from home.\n"
        "5. can be found at home.\n"
        "6. lives at home."
    ).format(n=n, prefix=prefix)
    return (
        "你是英文改写助手。任务:把下面英文句子里的 "
        f"「{target_word}」换成另一个英文词,使整句自然通顺。\n\n"
        "硬性规则:\n"
        f"  1. 「{target_word}」这个词必须被替换掉 —— 即使它是 be 动词、"
        "介词、连词这种『看起来非有不可』的词,你也得换一个。\n"
        f"  2. 你输出的每行,紧接「{prefix}」之后的第一个词,不能是 "
        f"「{target_word}」,必须是另一个英文词。\n"
        f"  3. 你只输出「{prefix}」之后的部分(不要重复「{prefix}」本身,"
        f"也不要写「{prefix}」之前的内容)。\n"
        "  4. 每行必须以句号、问号或感叹号结尾,是一段完整英文。\n"
        f"  5. 给出 {n} 个不同的版本,首词尽量多样化。\n"
        f"  6. 不要照抄示例里的句式或短语到自己答案里。\n\n"
        f"中文原文:{source_zh}\n"
        f"当前英文(完整):{current}\n"
        f"待替换的词:{target_word}\n"
        f"必须原样保留的前缀(不要重写、不要展开):{prefix}\n"
        f"「{prefix}」之后的内容(从这里开始改写,要改写到句末):{rest}\n\n"
        f"{few_shot}\n\n"
        "现在请按相同方式给出 "
        f"{n} 个不同的完整英文后缀,每个后缀第一个词不能是 「{target_word}」,"
        "每个后缀都以句号或问号或感叹号结尾:\n"
        + "\n".join(f"{i}." for i in range(1, n + 1))
    )


def _build_wordlist_prompt(
    source_zh: str,
    current: str,
    prefix: str,
    target_word: str,
    n: int,
) -> str:
    """兜底路径:让模型只给出 n 个替换词,代码侧做替换。"""
    rest = current
    pfx = prefix.strip()
    if pfx and rest.lower().startswith(pfx.lower()):
        rest = rest[len(pfx):].lstrip()
    return (
        "你是英文改写助手。任务:为下面句子里的「{tw}」列出 {n} 个"
        "可替换的英文单词或短语,使整句自然。\n\n"
        "硬性规则:\n"
        "  1. 你只输出 {n} 个替换项,每行一个,用 1. 2. 3. ... 编号。\n"
        "  2. 每个替换项不能是「{tw}」本身,也不能是它的大小写变化。\n"
        "  3. 如果语境里直接换单词会破坏语法,可以输出 1-3 个词的短语。\n"
        "  4. 不要写完整句子,只要替换项本身。\n\n"
        "中文原文:{src}\n"
        "当前英文:{cur}\n"
        "待替换的词:{tw}\n"
        "「{tw}」所在语境:{rest}\n\n"
        "示例(be 动词也要给替换项):\n"
        "中文:他在家\n"
        "当前:He is at home.\n"
        "待替换:is\n"
        "1. stays\n"
        "2. remains\n"
        "3. works from home\n"
        "4. spends time\n"
        "5. lives\n"
        "6. can be found\n\n"
        "现在请按相同方式列出 {n} 个替换项:\n"
        + "\n".join(f"{i}." for i in range(1, n + 1))
    ).format(tw=target_word, n=n, src=source_zh, cur=current, rest=rest)


# --- Parsing helpers --------------------------------------------------------


# 兼容 `1.` `1)` `1:` `①` `1、` 等大陆模型爱用的编号方式。
_NUM_PREFIX = re.compile(r"^\s*(?:\d+|[一二三四五六七八九十])[\.\)\:、]\s*")


def _strip_number(text: str) -> str:
    """剥掉前面的 ``1.`` / ``1)`` / ``1:`` 这种编号,顺便剥引导破折号/星号。"""
    s = text.strip()
    s = _NUM_PREFIX.sub("", s, count=1)
    # `- ` `* ` `• ` 开头常见于把列表翻译成 markdown 的情况。
    s = re.sub(r"^[-*•]\s+", "", s)
    return s.strip()


def _first_word(text: str) -> str:
    """剥去首部标点,返回第一个 token 的 lower-cased (无尾标点)。

    与 ``translator._first_word`` 语义对齐;不依赖它是为了让本脚本单独可跑。
    """
    s = text.lstrip().lstrip(",.;:!?\"'()[]{}")
    if not s:
        return ""
    return s.split(maxsplit=1)[0].lower().rstrip(".,!?;:\"")


def _head_after_prefix(text: str, prefix: str) -> str:
    """剥掉 prefix 后,真正「target 位置」上的 head word。"""
    rest = text
    pfx = prefix.strip()
    if pfx and rest.lower().startswith(pfx.lower()):
        rest = rest[len(pfx):].lstrip()
    return _first_word(rest)


def _ends_with_punct(s: str) -> bool:
    return bool(s) and s[-1] in ".!?"


# Few-shot 里出现的稳定短语;模型一旦把它们原样搬进真实任务输出,
# 就视作「模板泄漏」直接丢弃。视情况往里加。
_FEW_SHOT_LEAK_TAILS = (
    " our product",
    " our product.",
    "our product.",
)

# Few-shot A 给的 6 个改写,作为「完整短语」出现任一即视为模板泄漏。
# 真实任务里这些 1-3 词短语很少是合理改写(只配 "Let me ..." 这种小前缀),
# 出现大多是模型把示例当模板直接复用。
_FEW_SHOT_LEAK_BODIES = (
    "start now",
    "go",
    "proceed",
    "move on to the next step",
    "take action",
    "get going right away",
    "stays at home",
    "remains at home",
    "spends time at home",
    "works from home",
    "can be found at home",
    "lives at home",
)


def _looks_like_few_shot_leak(body: str, prefix: str = "") -> bool:
    """候选里如果出现 few-shot 里的固定短语尾巴,直接当模板泄漏处理。

    三种检测维度:
        1. 包含 ``_FEW_SHOT_LEAK_TAILS`` 里的尾巴(如 "our product");
        2. 整段去掉 prefix 之后,等于 ``_FEW_SHOT_LEAK_BODIES`` 里某项
           (few-shot 行的 1-1 复刻);
        3. 整段去掉 prefix 之后,跟某条 few-shot body 完全相同(忽略大小写、
           句末标点)。
    """
    b = body.lower()
    if any(tail in b for tail in _FEW_SHOT_LEAK_TAILS):
        return True
    # 剥掉 prefix 后再跟 few-shot body 比,否则 "The go." 会被判成 "the go",
    # 而 few-shot 里只有 "go",匹配不上。
    body_to_check = b
    pfx = prefix.strip().lower()
    if pfx and body_to_check.startswith(pfx):
        body_to_check = body_to_check[len(pfx):].lstrip()
    body_norm = body_to_check.rstrip(" .!?")
    if body_norm in _FEW_SHOT_LEAK_BODIES:
        return True
    return False


def _looks_like_repeated_subject(
    body: str, prefix: str, current: str
) -> bool:
    """启发式:如果 suffix 的开头出现「把 prefix 的内容又用别的词重写了一遍」
    这种「重述主语」的迹象,就丢掉。例:prefix="More than twenty people",
    suffix="several dozen participants attended the meeting."。
    特征:suffix 在前 ~6 个词里出现 prefix 的核心内容词("twenty"、"people")。
    """
    pfx_lower = prefix.lower()
    body_lower = body.lower()
    # 找 prefix 里长度 ≥ 4 的"内容词"作为锚点。
    content_words = [w for w in pfx_lower.split() if len(w.rstrip(".,!?;:")) >= 4]
    if not content_words:
        return False
    # 取出 body 头 6 个词作窗口。
    head_tokens = body_lower.split()[:6]
    head_blob = " ".join(head_tokens)
    # 至少有一个锚点词被重述,就视作模型在重写主语。
    return any(w.rstrip(".,!?;:") in head_blob for w in content_words)


def _is_length_bloated(body: str, prefix: str, current: str) -> bool:
    """suffix 不能比 current 在 prefix 之后的部分长 2 倍以上。

    模型偶尔在 prefix 后塞一大段自由发挥(像 case 9 的
    "several dozen participants" 之类),通过长度阈值把它们过滤掉。
    """
    rest = current
    pfx = prefix.strip()
    if pfx and rest.lower().startswith(pfx.lower()):
        rest = rest[len(pfx):].lstrip()
    return len(body) > 2 * max(1, len(rest)) + 20


def _split_candidates(
    raw: str, target_word: str, current: str, n: int, prefix: str = "",
) -> List[str]:
    """把模型原始输出切成最多 n 条候选 suffix。

    流程:
        1. 按行 split。
        2. 剥每行编号/破折号。
        3. 砍掉第一个编号之前的多余前言(模型有时先来一句
           "Here are 6 alternatives:")。
        4. 过滤:
           - head 词仍等于 target_word → 跳过;
           - 整段是 current 的当前 suffix → 跳过;
           - 整段以「prefix + target_word」开头 → 视为模型没听指令,
             砍掉前缀并接受(交给 rephrase_candidates 用 join_sentence 重拼);
           - 候选没有句末标点 → 当片段丢掉(模型只吐了「begin tonight」这种
             不完整表达);
           - 候选里含 few-shot 固定尾巴(如 "our product") → 模板泄漏丢掉;
           - 候选在 prefix 之后塞了重述主语(对应 case 9 那种
             "several dozen participants" 症状)→ 丢掉;
           - 候选长度 > 期望 suffix 的 2x + 20 字符 → 视作膨胀丢掉。
    """
    text = raw.strip()
    if not text:
        return []

    # Fast path: 模型一句话直接写完,不编号。
    if "\n" not in text:
        candidate = _strip_number(text)
        if not candidate or _first_word(candidate) == target_word.lower():
            return []
        if not _ends_with_punct(candidate):
            return []
        if _looks_like_few_shot_leak(candidate, prefix=prefix):
            return []
        if prefix and _looks_like_repeated_subject(candidate, prefix, current):
            return []
        if prefix and _is_length_bloated(candidate, prefix, current):
            return []
        return [candidate]

    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    start = 0
    for i, ln in enumerate(lines):
        if _NUM_PREFIX.match(ln) or re.match(r"^\s*[-*•]\s+", ln):
            start = i
            break
    lines = lines[start:]

    out: List[str] = []
    seen: set[str] = set()
    forbidden_head = _first_word(target_word)

    for ln in lines:
        body = _strip_number(ln)
        if not body:
            continue
        head = _first_word(body)
        if not head:
            continue
        if head == forbidden_head:
            continue
        # 关键:即使 head != target,模型有时会「完整句」开头包含 prefix
        # (没听 suffix-only 指令),此时剥掉 prefix 后的真正 head 才应该
        # 被检查。这个检查对应 case 5 的 "We use tools..." 那种症状。
        if prefix:
            pfx_lc = prefix.strip().lower()
            if body.lower().startswith(pfx_lc):
                rest_after = body[len(prefix.strip()):].lstrip()
                head_after = _first_word(rest_after)
                if head_after == forbidden_head:
                    continue
        if not _ends_with_punct(body):
            continue
        if _looks_like_few_shot_leak(body, prefix=prefix):
            continue
        if prefix and _looks_like_repeated_subject(body, prefix, current):
            continue
        if prefix and _is_length_bloated(body, prefix, current):
            continue
        key = body.lower().rstrip(" .!?")
        if key in seen:
            continue
        seen.add(key)
        out.append(body)
        if len(out) >= n:
            break

    return out


def _split_replacement_words(
    raw: str, target_word: str, n: int
) -> List[str]:
    """解析「单词/短语」路径的模型输出,只留不含 target 本身的短词或短语。

    接受任意长度(0 到 ~3 词),但滤掉:
        - 含 target 的整段(说明模型没听指令,复制粘贴了原 target);
        - 太长的(超过 3 个 token,大概率不是「替换词」而是另一句完整句);
        - 重复、空行。
    """
    text = raw.strip()
    if not text:
        return []
    target_lc = target_word.lower().rstrip(" .,!?;:")

    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    start = 0
    for i, ln in enumerate(lines):
        if _NUM_PREFIX.match(ln) or re.match(r"^\s*[-*•]\s+", ln):
            start = i
            break
    lines = lines[start:]

    out: List[str] = []
    seen: set[str] = set()
    for ln in lines:
        body = _strip_number(ln).strip().rstrip(" .!?,;:")
        if not body:
            continue
        # 太长就当不是替换词(超过 4 个 token 通常是句子)。
        if len(body.split()) > 4:
            continue
        # 整段就是 target 的某种形态 → 跳过。
        if body.lower() == target_lc:
            continue
        # 头词仍是 target 的形态 → 跳过(说明模型没换)。
        if _first_word(body) == target_lc:
            continue
        # 整段照搬 prefix / current → 跳过(同前)。
        if body.lower() in seen:
            continue
        seen.add(body.lower())
        out.append(body)
        if len(out) >= n:
            break
    return out


def _substitute_target(current: str, target_word: str, replacement: str) -> str:
    """把 ``current`` 里的 ``target_word`` 替换成 ``replacement``。

    按整词匹配(基于 ``\b`` 边界)。大小写不敏感,但保持原 target 的
    大小写形态(只把 target 词位改成 replacement 自身的大小写)。
    """
    pattern = re.compile(r"\b" + re.escape(target_word) + r"\b", re.IGNORECASE)

    def _replace_one(m: re.Match) -> str:
        original = m.group(0)
        # 大小写保留:全大写就全大写,首字母大写就首字母大写,否则照搬。
        if original.isupper():
            return replacement.upper()
        if original[0].isupper():
            return replacement[:1].upper() + replacement[1:]
        return replacement

    return pattern.sub(_replace_one, current, count=1)


# --- Ollama I/O -------------------------------------------------------------


class ModelNotAvailableError(RuntimeError):
    """ollama 进程起不来、模型未 pull 或网络不通。


    报清楚原因,而不是抛 ollama 库的原始堆栈。"""


def _check_model_available(client: "ollama.Client", model: str) -> None:
    """确认模型已 pull。失败时抛 ``ModelNotAvailableError`` 给前端漂亮错。"""
    try:
        listed = client.list()
    except Exception as e:  # noqa: BLE001
        raise ModelNotAvailableError(
            f"无法连接 ollama 进程(默认 http://127.0.0.1:11434)。错误:{e}"
        ) from e
    # ollama 0.6 list() 返回 ``ListResponse(models=[Model(model=..., ...), ...])``;
    # 再老一点的版本可能返回 ``{"models": [{...}, ...]}``(字典)。
    names: set[str] = set()
    models_attr = getattr(listed, "models", None)
    if models_attr is None and isinstance(listed, dict):
        models_attr = listed.get("models", [])
    for item in models_attr or []:
        # ``Model`` 对象有 ``.model`` 字段;字典可能有 ``name`` / ``model``。
        if isinstance(item, dict):
            for key in ("model", "name"):
                val = item.get(key)
                if val:
                    names.add(val)
        else:
            for attr in ("model", "name"):
                val = getattr(item, attr, None)
                if val:
                    names.add(str(val))
                    break
    base = model.split(":", 1)[0]
    if model not in names and base not in {n.split(":", 1)[0] for n in names}:
        raise ModelNotAvailableError(
            f"ollama 中找不到模型 \"{model}\"。\n"
            f"  本地有:{sorted(n for n in names if n) or '（空）'}\n"
            f"  请先 ollama pull {model}"
        )


def _call_ollama(
    prompt: str,
    *,
    model: str,
    base_url: Optional[str],
    temperature: float,
) -> str:
    """单次调 ollama chat,返回 ``response.message.content`` 字符串。"""
    kwargs = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    if base_url:
        kwargs["host"] = base_url
    # ollama python client 用 ``options`` 传推理参数。
    kwargs["options"] = {"temperature": temperature}
    resp: ChatResponse = chat(**kwargs)
    return resp.message.content or ""


# --- Translation ------------------------------------------------------------


def translate_zh_to_en(
    source_zh: str,
    *,
    model: str = DEFAULT_MODEL,
    base_url: Optional[str] = None,
    debug: bool = False,
) -> str:
    """用 ollama 模型把中文翻成英文。

    hy-mt2:1.8b 是腾讯的 CN→EN 翻译模型,直接给「请翻译 + 只输出结果」的
    中文指令就行,实测 ``temperature=0.0`` 下输出稳定单句、不会复述中文、
    不会加引号或额外说明。

    返回清洗后的英文整句(去前后空白、引号包裹去掉、换行合并成空格)。
    抛 ``ModelNotAvailableError`` / ``ValueError``。
    """
    if not (source_zh and source_zh.strip()):
        raise ValueError("source_zh 不能为空。")
    prompt = (
        "请把下面的中文翻译成英文,只输出翻译结果,不要解释、不要重复中文:\n"
        f"{source_zh.strip()}"
    )
    raw = _call_ollama(prompt, model=model, base_url=base_url, temperature=0.0)
    if debug:
        sys.stderr.write(f"\n[debug] ── translate raw ──\n{raw}\n")
        sys.stderr.flush()
    cleaned = _clean_translation(raw)
    if not cleaned:
        raise RuntimeError(f"模型返回的翻译为空:raw={raw!r}")
    return cleaned


def _clean_translation(raw: str) -> str:
    """清洗 ollama 翻译输出。

    1. 整段 strip。
    2. 合并换行成空格(模型偶发多行)。
    3. 剥最外层成对引号(单/双)。
    4. 剥前缀闲话,如 "Translation:"、"英文:".
    5. 若模型在前面回吐了一遍中文(以常见中文标点结尾),砍掉那段。
    """
    s = raw.strip()
    if not s:
        return ""
    # 多行合并(用空格连接,保持自然的英文断句)。
    s = re.sub(r"\s*\n\s*", " ", s)
    s = s.strip()
    # 剥成对外引号(单/双)。
    for q in ('"""', "'''", '"', "'"):
        if len(s) >= 2 * len(q) and s.startswith(q) and s.endswith(q):
            s = s[len(q) : -len(q)].strip()
            break
    # 剥前缀闲话(冒号结尾的小尾巴)。
    s = re.sub(
        r"^(?:translation|translated|english|英文|译|翻译|以下是|结果|answer|output)\s*[:：]\s*",
        "",
        s,
        flags=re.IGNORECASE,
    ).strip()
    # 如果开头是中文(以中文标点或汉字结尾的句子),砍掉那段。
    # 例: "您的反馈将影响我们。 Your feedback will..." → 只留英文。
    m = re.match(
        r"^([一-鿿][一-鿿\s,。!?、:;()()《》\"""'']*?[一-鿿,。!?、:;()()《》\"""''])\s+([A-Za-z].*)$",
        s,
    )
    if m:
        s = m.group(2).strip()
    return s


# --- Public API -------------------------------------------------------------


def rephrase_candidates(
    source_zh: str,
    current: str,
    prefix: str,
    target_word: str,
    n: int = 6,
    *,
    model: str = DEFAULT_MODEL,
    base_url: Optional[str] = None,
    debug: bool = False,
) -> List[str]:
    """调用本地 ollama 模型,返回最多 n 条完整英文改写候选。

    每条候选必须满足:
        - 全文以 ``prefix`` (忽略大小写) 开头。
        - 紧跟 prefix 之后的第一个词 ≠ ``target_word`` (忽略大小写与尾标点)。
        - 与 ``current`` 不完全相同。

    若第一次调用解析不出 ``n`` 条、且 ``debug=False``,自动 ``temperature=0.7``
    再试一次。若仍失败,原样返回已有候选(可能少于 ``n``)。

    抛出:
        ModelNotAvailableError -- ollama 进程不可达或模型不存在。
        ValueError -- 4 个核心入参任一为空。
    """
    if not (source_zh and current and prefix and target_word):
        raise ValueError(
            "source_zh / current / prefix / target_word 都不能为空。"
        )
    if n < 1:
        raise ValueError("n 必须 ≥ 1")

    # 延迟导入 client,只为提前做可用性检查。
    import ollama as _ollama  # type: ignore[import-not-found]

    if base_url:
        client = _ollama.Client(host=base_url)
    else:
        client = _ollama.Client()
    _check_model_available(client, model)

    prompt = _build_prompt(source_zh, current, prefix, target_word, n)

    if debug:
        sys.stderr.write("\n[debug] ── prompt ──\n")
        sys.stderr.write(prompt)
        sys.stderr.write("\n")
        sys.stderr.flush()

    raw = _call_ollama(prompt, model=model, base_url=base_url, temperature=0.2)
    if debug:
        sys.stderr.write("\n[debug] ── raw response ──\n")
        sys.stderr.write(raw)
        sys.stderr.write("\n")
        sys.stderr.flush()

    out = _split_candidates(raw, target_word, current, n, prefix=prefix)

    # 触发补一次重试的两种情况:
    #   (a) 解析出的合格候选数过少(明显的取数不够);
    #   (b) 原始输出里看到 few-shot 模板泄漏的迹象(出现 "our product" 等)
    #       —— 这种就算解析出 6 条,本质也是错的全要丢,不如直接重抽。
    leak_in_raw = _looks_like_few_shot_leak(raw, prefix=prefix)
    if len(out) < max(1, n // 2) or leak_in_raw:
        raw2 = _call_ollama(prompt, model=model, base_url=base_url, temperature=0.7)
        if debug:
            sys.stderr.write("\n[debug] ── retry raw response ──\n")
            sys.stderr.write(raw2)
            sys.stderr.write("\n")
            sys.stderr.flush()
        # 重试结果直接覆写 `out`,避免把被泄漏污染的「旧候选」混进最终列表。
        out = _split_candidates(raw2, target_word, current, n, prefix=prefix)

    # 兜底:主路径解析不出 n 条合格候选,改走「只让模型给替换词」的路径。
    # 这一步保证 head != target(由替换动作强制生效)与语法基本合理(模型
    # 选词时已经把上下文喂给它了)。
    if len(out) < n:
        wl_prompt = _build_wordlist_prompt(source_zh, current, prefix, target_word, n)
        if debug:
            sys.stderr.write("\n[debug] ── wordlist prompt ──\n")
            sys.stderr.write(wl_prompt)
            sys.stderr.write("\n")
        raw_wl = _call_ollama(wl_prompt, model=model, base_url=base_url, temperature=0.4)
        if debug:
            sys.stderr.write("\n[debug] ── wordlist raw ──\n")
            sys.stderr.write(raw_wl)
            sys.stderr.write("\n")
            sys.stderr.flush()
        words = _split_replacement_words(raw_wl, target_word, n)
        # 用「替换词」直接对 current 做整词替换 → 自动得一批 head != target
        # 且语法继承自原句的候选。
        for w in words:
            out.append(_substitute_target(current, target_word, w))

    # 模型有时把 prefix 也写进 suffix(没听指令),把前导 prefix 砍掉再拼接。
    # 之后还得再确认「紧跟 prefix 的 head」≠ target —— 这一道防御是兜底
    # _split_candidates 漏过的(例如 fallback 路径直接喂完整句的情况)。
    pfx_lc = prefix.strip().lower()
    forbidden_head = _first_word(target_word)
    full: List[str] = []
    seen_full: set[str] = set()
    cur_full_lc = current.strip().lower()
    for suffix in out:
        sfx = suffix.strip()
        if pfx_lc and sfx.lower().startswith(pfx_lc):
            sfx = sfx[len(prefix.strip()):].lstrip()
        sentence = _join_prefix(prefix, sfx)
        # 严格约束 1:head (即 prefix 之后第一个词) 不能是 target。
        rest_after = sentence[len(prefix):].lstrip() if sentence.lower().startswith(pfx_lc) else ""
        if rest_after and _first_word(rest_after) == forbidden_head:
            continue
        # 去重:同一完整句子只保留一份,且不能跟 current 完全一样。
        key = sentence.lower().rstrip(" .!?")
        if not key or key == cur_full_lc.rstrip(" .!?") or key in seen_full:
            continue
        seen_full.add(key)
        full.append(sentence)
        if len(full) >= n:
            break
    return full


def _join_prefix(prefix: str, suffix: str) -> str:
    """代码侧 prepend prefix。

    与 ``translator.join_sentence`` 同义,但 self-contained,避免单脚本外依赖。
    """
    if not prefix:
        return suffix
    # 后缀若以闭合标点开头(, . ! ? ; : ) ] }),用 ``prefix + suffix`` 不留空格;
    # 否则 ``prefix + ' ' + suffix``。
    if suffix and suffix[0] in (",", ".", "!", "?", ";", ":", ")", "]", "}"):
        return f"{prefix}{suffix}"
    return f"{prefix} {suffix}"


# --- CLI --------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ollama_rephrase",
        description="用本地 ollama 模型给英文句子的某个位置提供改写候选。",
    )
    p.add_argument("--source", required=True, help="中文原句。")
    p.add_argument("--current", required=True, help="当前英文整句。")
    p.add_argument("--prefix", required=True, help="需要原样保留的前缀。")
    p.add_argument("--target", required=True, help="要被替换掉的词。")
    p.add_argument(
        "-n",
        "--num",
        type=int,
        default=6,
        help="希望的候选数。默认 6,范围 4-8。",
    )
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"ollama 模型名。默认 {DEFAULT_MODEL}。",
    )
    p.add_argument(
        "--ollama-url",
        default=None,
        help="可选:自定义 ollama 服务地址,例如 http://127.0.0.1:11434。",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="把 prompt + 模型原始回复打到 stderr。",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="输出 JSON 数组而非编号文本(便于脚本拼接)。",
    )
    return p


def _format_output(candidates: List[str], as_json: bool) -> str:
    if as_json:
        return json.dumps(candidates, ensure_ascii=False, indent=2)
    if not candidates:
        return ""
    width = max(2, len(str(len(candidates))))
    return "\n".join(f"{i:>{width}}. {c}" for i, c in enumerate(candidates, 1))


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    n = max(1, min(8, args.num))
    try:
        cands = rephrase_candidates(
            source_zh=args.source.strip(),
            current=args.current.strip(),
            prefix=args.prefix.strip(),
            target_word=args.target.strip(),
            n=n,
            model=args.model,
            base_url=args.ollama_url,
            debug=args.debug,
        )
    except ModelNotAvailableError as e:
        sys.stderr.write(f"ERROR: {e}\n")
        return 1
    except ValueError as e:
        sys.stderr.write(f"ERROR: {e}\n")
        return 2

    if not cands:
        sys.stderr.write(
            "ERROR: 模型返回了 0 条合格候选(可能受 --target/--prefix 限制,"
            "尝试 --debug 看原始输出并放宽约束)。\n"
        )
        # 仍然在 stdout 打印空,让调用方好判断。
        print(_format_output([], args.json))
        return 1

    print(_format_output(cands, args.json))
    return 0


# --- OllamaTranslator -------------------------------------------------------


class OllamaTranslator:
    """``translator.Translator`` 的 ollama 等价物。

    对外接口刻意对齐:

    - ``.translate(source_zh: str) -> str``
    - ``.rephrase(source_zh, current, prefix, target_word) -> list[str]``

    这样 ``interactive_translate.py`` 之外的代码(如 ``server.py``) 以后想
    换后端,只要把 ``Translator(...)`` 换成 ``OllamaTranslator(...)`` 即可。

    构造时立刻做 ollama 可达性 + 模型存在性检查,避免到第一次调用才发现
    模型没 pull。抛 ``ModelNotAvailableError``。
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        base_url: Optional[str] = None,
        debug: bool = False,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.debug = debug
        import ollama as _ollama  # type: ignore[import-not-found]

        if base_url:
            client = _ollama.Client(host=base_url)
        else:
            client = _ollama.Client()
        _check_model_available(client, model)

    def translate(self, source_zh: str) -> str:
        return translate_zh_to_en(
            source_zh, model=self.model, base_url=self.base_url, debug=self.debug,
        )

    def rephrase(
        self,
        source_zh: str,
        current: str,
        prefix: str,
        target_word: str,
    ) -> list[str]:
        return rephrase_candidates(
            source_zh=source_zh,
            current=current,
            prefix=prefix,
            target_word=target_word,
            model=self.model,
            base_url=self.base_url,
            debug=self.debug,
        )


if __name__ == "__main__":
    raise SystemExit(main())
