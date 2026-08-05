#!/usr/bin/env ~/.pyenv/versions/qlib/bin/python
"""v3 JSON 后处理：去重 + 重平衡到 3-5 词/组。"""
import json
import sys
from pathlib import Path

PATH = Path("/Users/huhao/src/codesnip/python/ai/021-cet4/cet4_hexinci_groups_v3.json")


def greedy_chunk_3_to_5(words: list[str]) -> list[list[str]]:
    n = len(words)
    if n < 3:
        return [list(words)] if n > 0 else []
    result: list[list[str]] = []
    i = 0
    while i < n:
        remaining = n - i
        if remaining in (3, 4, 5):
            result.append(words[i:i + remaining])
            i += remaining
        elif remaining in (1, 2):
            if result:
                prev = result[-1]
                need = 3 - remaining
                take = min(need, len(prev))
                moved = prev[-take:]
                del prev[-take:]
                merged = moved + words[i:]
                result.append(merged)
            else:
                result.append(words[i:])
            i += remaining
        else:  # remaining >= 6
            result.append(words[i:i + 4])
            i += 4
    return result


def fix_groups(groups: list[dict]) -> tuple[list[dict], int]:
    """去重 + 收集遗漏词，返回修复后的 groups 和遗漏词列表。"""
    seen: set[str] = set()
    cleaned: list[dict] = []
    for g in groups:
        new_words = []
        for w in g["words"]:
            if w not in seen:
                seen.add(w)
                new_words.append(w)
        if new_words:
            g2 = dict(g)
            g2["words"] = new_words
            cleaned.append(g2)
    return cleaned, []  # 没有遗漏词（去重不影响总数）


def main() -> int:
    db = json.loads(PATH.read_text(encoding="utf-8"))

    total_before = sum(len(g["words"]) for cat in db["categories"] for g in cat["groups"])
    print(f"修复前组内总词数: {total_before}")

    # Step 1: 对每个 cat 的 groups 去重（首现保留）
    for cat in db["categories"]:
        cat["groups"], _ = fix_groups(cat["groups"])

    total_after_dedup = sum(len(g["words"]) for cat in db["categories"] for g in cat["groups"])
    print(f"去重后组内总词数: {total_after_dedup}")
    print(f"  减少: {total_before - total_after_dedup} 个重复")

    # Step 2: 校验哪些组 < 3 词
    bad = []
    for cat in db["categories"]:
        for g in cat["groups"]:
            if not (3 <= len(g["words"]) <= 5):
                bad.append((cat, g))
    if bad:
        print(f"\n发现 {len(bad)} 个 < 3 词 或 > 5 词 的组，重平衡：")
        for cat, g in bad:
            print(f"  cat {cat['category_id']} {cat['name_zh']} g{g['group_id']} ({g.get('title_zh','')}): {len(g['words'])} 词 {g['words']}")

    # Step 3: 全 cat 范围内重平衡：把 < 3 词的组拆出词，合并到其他组
    # Strategy: 收集所有 < 3 词的组的"短词"，均摊到 < 5 词的组里
    for cat in db["categories"]:
        # 反复扫描直到所有组都 3-5 词
        for _ in range(50):  # 上限 50 次防死循环
            short_groups = [(i, g) for i, g in enumerate(cat["groups"]) if len(g["words"]) < 3]
            long_groups  = [(i, g) for i, g in enumerate(cat["groups"]) if len(g["words"]) > 5]
            if not short_groups and not long_groups:
                break
            # 处理 long groups: 拆
            for i, g in long_groups:
                ws = g["words"]
                while len(ws) > 5:
                    new = {"title_zh": g["title_zh"], "title_en": g.get("title_en",""), "words": ws[:4], "explanation_zh": g.get("explanation_zh","")}
                    cat["groups"].append(new)
                    ws = ws[4:]
                g["words"] = ws
            # 处理 short groups: 把它的词移到最后一个 < 5 词的组
            for i, g in short_groups:
                if not g["words"]:
                    continue
                # 找目标：< 5 词的组（最后一个）
                target = next((g2 for g2 in reversed(cat["groups"]) if 3 <= len(g2["words"]) < 5), None)
                if target:
                    target["words"].extend(g["words"])
                else:
                    # 兜底：找任意 < 5 词的组
                    target = next((g2 for g2 in cat["groups"] if len(g2["words"]) < 5), None)
                    if target:
                        target["words"].extend(g["words"])
                g["words"] = []  # 清空（稍后统一删除）

        # 删除空组
        cat["groups"] = [g for g in cat["groups"] if g["words"]]

        # 最终兜底：如果还有 < 3 词的组，贪心重切
        for _ in range(10):
            all_words = []
            for g in cat["groups"]:
                all_words.extend(g["words"])
            sizes = [len(g["words"]) for g in cat["groups"]]
            if all(3 <= s <= 5 for s in sizes):
                break
            # 重新切
            blocks = greedy_chunk_3_to_5(all_words)
            cat["groups"] = [
                {"title_zh": g.get("title_zh",""), "title_en": g.get("title_en",""), "words": blk, "explanation_zh": g.get("explanation_zh","")}
                for g, blk in zip(cat["groups"], blocks)
            ]

    # 重新分配 group_id
    next_gid = 1
    for cat in db["categories"]:
        for g in cat["groups"]:
            g["group_id"] = next_gid
            next_gid += 1
    db["total_groups"] = next_gid - 1

    # 更新 total_words_grouped
    all_words = [w for cat in db["categories"] for g in cat["groups"] for w in g["words"]]
    db["total_words_grouped"] = len(set(all_words))

    # 终极校验
    print(f"\n--- 修复后 ---")
    print(f"总词数（去重后）: {len(set(all_words))}")
    print(f"组数: {db['total_groups']}")

    vocab = {l.strip() for l in open("/Users/huhao/src/codesnip/python/ai/021-cet4/cet4_sijizhenti_hexinci.txt") if l.strip()}
    missing = vocab - set(all_words)
    extra   = set(all_words) - vocab
    if missing:
        print(f"⚠️  缺失 {len(missing)} 词: {sorted(missing)[:10]}")
        db["ungrouped_words"] = sorted(missing)
    else:
        db["ungrouped_words"] = []
    if extra:
        print(f"⚠️  多出 {len(extra)} 词表外词: {sorted(extra)[:10]}")
    print(f"ungrouped_words: {len(db['ungrouped_words'])}")

    bad = []
    for cat in db["categories"]:
        for g in cat["groups"]:
            if not (3 <= len(g["words"]) <= 5):
                bad.append((cat["category_id"], g["group_id"], len(g["words"])))
    if bad:
        print(f"⚠️  仍有 {len(bad)} 个组词数不合规: {bad[:5]}")
    else:
        print(f"✓ 全部 {db['total_groups']} 组词数 3-5 合规")

    # 写回
    PATH.write_text(json.dumps(db, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n💾 已写回: {PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
