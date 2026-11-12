#!/usr/bin/env ~/.pyenv/versions/qlib/bin/python
"""v3 补讲解：对缺讲解的组重跑 pass2b。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import hexinci_group_v3 as h

PATH = Path("/Users/huhao/src/codesnip/python/ai/021-cet4/cet4_hexinci_groups_v3.json")


def main() -> int:
    db = json.loads(PATH.read_text(encoding="utf-8"))
    client = h.anthropic.Anthropic()

    # 找所有缺讲解的组，按 cat 分组
    by_cat: dict[int, tuple[dict, list[dict]]] = {}
    for cat in db["categories"]:
        missing = [g for g in cat["groups"]
                   if not g.get("explanation_zh")
                   or "auto-fix" in g.get("explanation_zh", "")
                   or "待补" in g.get("explanation_zh", "")
                   or "auto-补到末组" in g.get("explanation_zh", "")]
        if missing:
            by_cat[cat["category_id"]] = (cat, missing)

    print(f"发现 {sum(len(g) for _, g in by_cat.values())} 个组缺讲解，分布在 {len(by_cat)} 个大类")

    # 复用 incremental.py 的 args 风格
    class Args:
        debug = True
        max_retries = 3
        model = "MiniMax-M3"
        sleep = 0

    for cid, (cat, missing_groups) in by_cat.items():
        print(f"\n▶ 补 cat {cid} {cat['name_zh']} 的 {len(missing_groups)} 组讲解")
        # 清空占位符（pass2b 的 pending 判定需要 explanation 为空或含 "auto-fix"）
        for g in missing_groups:
            g["explanation_zh"] = ""
        # 直接用 pass2b 函数
        h.pass2b_fill_explanations_for_groups(
            client, missing_groups, cat["name_zh"], cat.get("name_en", ""), args=Args(),
        )
        # 写回 db
        gid_set = {g["group_id"] for g in missing_groups}
        for i, g in enumerate(cat["groups"]):
            if g["group_id"] in gid_set:
                cat["groups"][i] = missing_groups.pop(0)
                if not missing_groups:
                    break

    # 写回
    PATH.write_text(json.dumps(db, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n💾 已写回: {PATH}")

    # 再次校验
    no_expl = sum(1 for c in db["categories"] for g in c["groups"]
                  if not g.get("explanation_zh")
                  or "auto-fix" in g.get("explanation_zh", "")
                  or "待补" in g.get("explanation_zh", "")
                  or "auto-补到末组" in g.get("explanation_zh", ""))
    print(f"补完后仍缺讲解: {no_expl} 组")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
