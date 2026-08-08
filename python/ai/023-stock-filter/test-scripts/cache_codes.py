"""eltdx 代码表本地缓存。

策略:
- 全量拉取 sh/sz/bj 三个市场的 SecurityCode
- 以 JSON 持久化(eltdx 的 SecurityCode 自带 to_jsonable)
- 默认 TTL = 24h;超过则下次调用时强制刷新
- 提供按板块/品种筛选的派生视图
- 附带按题材板块筛选(by-topic),seed 概念见 docs/cache_codes.md §4.8

用法:
    ~/.pyenv/versions/qlib/bin/python test-scripts/cache_codes.py refresh
    ~/.pyenv/versions/qlib/bin/python test-scripts/cache_codes.py list --board sse_star_market
    ~/.pyenv/versions/qlib/bin/python test-scripts/cache_codes.py list --category a_share
    ~/.pyenv/versions/qlib/bin/python test-scripts/cache_codes.py lookup sh688825
    ~/.pyenv/versions/qlib/bin/python test-scripts/cache_codes.py stats
    ~/.pyenv/versions/qlib/bin/python test-scripts/cache_codes.py topics --seed sh688825
    ~/.pyenv/versions/qlib/bin/python test-scripts/cache_codes.py by-topic 存储芯片 --seed sh688825
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from eltdx import TdxClient, to_jsonable

CACHE_PATH = Path(__file__).resolve().parent.parent / ".cache" / "eltdx_codes.json"
TOPIC_CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "topics"
TOPICS_SEED_CACHE = Path(__file__).resolve().parent.parent / ".cache" / "topics_seed.json"
TTL_SECONDS = 24 * 3600
TOPIC_TTL_SECONDS = 6 * 3600  # 题材成分股变动较快,TTL 短一些
DEFAULT_SEED_CODE = "sz000001"  # 平安银行,题材覆盖面广


@dataclass
class CodesCache:
    """带 TTL 的代码表缓存。"""

    path: Path = CACHE_PATH
    ttl: int = TTL_SECONDS

    def is_fresh(self) -> bool:
        if not self.path.exists():
            return False
        age = time.time() - self.path.stat().st_mtime
        return age < self.ttl

    def load(self) -> dict:
        with self.path.open(encoding="utf-8") as f:
            return json.load(f)

    def save(self, codes: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "fetched_at": time.time(),
            "fetched_at_iso": time.strftime("%Y-%m-%d %H:%M:%S"),
            "count": len(codes),
            "codes": codes,
        }
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def fetch_remote(self) -> list[dict]:
        """连接真实主站,拉取三个市场的全量代码表。"""
        all_codes: list[dict] = []
        with TdxClient(timeout=5) as client:
            for market in ("sh", "sz", "bj"):
                items = client.get_codes_all(market)
                for item in items:
                    d = to_jsonable(item)
                    # full_code 是 property,JSON 化时丢失,需要补回来
                    d["full_code"] = item.full_code
                    all_codes.append(d)
        return all_codes

    def ensure(self, force: bool = False) -> dict:
        """按需刷新,返回当前缓存字典。"""
        if force or not self.is_fresh():
            print(f"[cache] refreshing from remote...", file=sys.stderr)
            codes = self.fetch_remote()
            self.save(codes)
            return self.load()
        return self.load()


class TopicCache:
    """题材成分股缓存。

    文件组织:
      .cache/topics_seed.json   — 指定 seed_code 的题材目录 {id: {name, ...}}
      .cache/topics/<seed>__<topic_id>.json  — 单个题材的成分股快照
    """

    def __init__(self, ttl: int = TOPIC_TTL_SECONDS) -> None:
        self.ttl = ttl
        TOPIC_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _seed_path(seed_code: str) -> Path:
        return TOPICS_SEED_CACHE

    @staticmethod
    def _topic_path(seed_code: str, topic_id: str | int) -> Path:
        return TOPIC_CACHE_DIR / f"{seed_code}__{topic_id}.json"

    def _is_seed_fresh(self, seed_code: str) -> bool:
        if not TOPICS_SEED_CACHE.exists():
            return False
        age = time.time() - TOPICS_SEED_CACHE.stat().st_mtime
        return age < self.ttl

    def load_seed(self, seed_code: str, force: bool = False) -> list[dict]:
        """拉取某 seed_code 可访问的全部题材。"""
        if not force and self._is_seed_fresh(seed_code):
            with TOPICS_SEED_CACHE.open(encoding="utf-8") as f:
                payload = json.load(f)
            if payload.get("seed_code") == seed_code:
                return payload["topics"]

        print(f"[topics] fetching seed={seed_code}...", file=sys.stderr)
        with TdxClient(timeout=5) as client:
            resp = client.f10.hot_topics(seed_code)
        topics = resp.first_table.rows

        payload = {
            "seed_code": seed_code,
            "fetched_at": time.time(),
            "fetched_at_iso": time.strftime("%Y-%m-%d %H:%M:%S"),
            "topics": topics,
        }
        with TOPICS_SEED_CACHE.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return topics

    def load_topic(self, seed_code: str, topic_id: str | int, force: bool = False) -> dict:
        path = self._topic_path(seed_code, topic_id)
        if not force and path.exists() and time.time() - path.stat().st_mtime < self.ttl:
            with path.open(encoding="utf-8") as f:
                return json.load(f)

        print(f"[topics] fetching topic {topic_id} via seed={seed_code}...", file=sys.stderr)
        with TdxClient(timeout=5) as client:
            resp = client.f10.topic_compare(seed_code, str(topic_id))
        rows = resp.first_table.rows

        payload = {
            "seed_code": seed_code,
            "topic_id": str(topic_id),
            "fetched_at": time.time(),
            "fetched_at_iso": time.strftime("%Y-%m-%d %H:%M:%S"),
            "count": len(rows),
            "stocks": rows,
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return payload

    def resolve_topic(
        self,
        name_or_id: str,
        seed_code: str = DEFAULT_SEED_CODE,
        force: bool = False,
    ) -> tuple[str, str]:
        """按名称或 ID 解析题材,返回 (topic_id, topic_name)。"""
        topics = self.load_seed(seed_code, force=force)
        # 1) 尝试按 ID 精确匹配
        for t in topics:
            if str(t.get("id")) == str(name_or_id):
                return str(t["id"]), t["ztmc"]
        # 2) 按名称精确匹配
        for t in topics:
            if t.get("ztmc") == name_or_id:
                return str(t["id"]), t["ztmc"]
        # 3) 按名称包含模糊匹配
        matches = [t for t in topics if name_or_id in (t.get("ztmc") or "")]
        if len(matches) == 1:
            return str(matches[0]["id"]), matches[0]["ztmc"]
        if matches:
            print(
                f"[topics] ambiguous match for '{name_or_id}', candidates:",
                file=sys.stderr,
            )
            for m in matches:
                print(f"  - id={m['id']:<6} name={m['ztmc']}", file=sys.stderr)
            raise SystemExit(2)
        raise SystemExit(f"[topics] no topic matched: {name_or_id}")


def filter_codes(
    data: dict,
    *,
    board: str | None = None,
    category: str | None = None,
    exchange: str | None = None,
    name_contains: str | None = None,
) -> list[dict]:
    rows = data["codes"]
    if board:
        rows = [r for r in rows if r["board"] == board]
    if category:
        rows = [r for r in rows if r["category"] == category]
    if exchange:
        rows = [r for r in rows if r["exchange"] == exchange]
    if name_contains:
        needle = name_contains.lower()
        rows = [r for r in rows if needle in r["name"].lower()]
    return rows


def cmd_refresh(args: argparse.Namespace) -> None:
    cache = CodesCache()
    cache.ensure(force=True)
    data = cache.load()
    print(f"[refresh] saved {data['count']} codes to {cache.path}")


def cmd_list(args: argparse.Namespace) -> None:
    cache = CodesCache()
    data = cache.ensure(force=args.force)
    rows = filter_codes(
        data,
        board=args.board,
        category=args.category,
        exchange=args.exchange,
        name_contains=args.name,
    )
    print(f"# total={data['count']}, matched={len(rows)}")
    print(f"{'full_code':<12} {'name':<14} {'board':<22} {'category':<10} {'prev_close':>10}")
    for r in rows[: args.limit]:
        print(
            f"{r['full_code']:<12} {r['name']:<14} {r['board']:<22} "
            f"{r['category']:<10} {r['previous_close_price']:>10.2f}"
        )
    if len(rows) > args.limit:
        print(f"... ({len(rows) - args.limit} more)")


def cmd_lookup(args: argparse.Namespace) -> None:
    cache = CodesCache()
    data = cache.ensure(force=args.force)
    target = args.code.lower()
    for r in data["codes"]:
        if r["full_code"].lower() == target or r["code"] == args.code:
            print(json.dumps(r, ensure_ascii=False, indent=2))
            return
    print(f"not found: {args.code}", file=sys.stderr)
    sys.exit(1)


def cmd_stats(args: argparse.Namespace) -> None:
    cache = CodesCache()
    data = cache.ensure(force=args.force)
    print(f"path        : {cache.path}")
    print(f"fetched_at  : {data['fetched_at_iso']}")
    print(f"ttl_seconds : {cache.ttl}")
    print(f"fresh       : {cache.is_fresh()}")
    print(f"total_codes : {data['count']}")

    # 按 exchange 统计
    by_exchange: {str, int} = {}
    by_board: {str, int} = {}
    by_category: {str, int} = {}
    for r in data["codes"]:
        by_exchange[r["exchange"]] = by_exchange.get(r["exchange"], 0) + 1
        by_board[r["board"]] = by_board.get(r["board"], 0) + 1
        by_category[r["category"]] = by_category.get(r["category"], 0) + 1

    print("\n[by exchange]")
    for k, v in sorted(by_exchange.items()):
        print(f"  {k:<4} {v:>5}")
    print("\n[by board]")
    for k, v in sorted(by_board.items(), key=lambda x: -x[1]):
        print(f"  {k:<24} {v:>5}")
    print("\n[by category]")
    for k, v in sorted(by_category.items(), key=lambda x: -x[1]):
        print(f"  {k:<12} {v:>5}")


def cmd_topics(args: argparse.Namespace) -> None:
    """列出 seed_code 可访问的全部题材。"""
    tcache = TopicCache()
    topics = tcache.load_seed(args.seed, force=args.force)
    print(f"# seed={args.seed}, total={len(topics)}")
    print(f"{'id':<8} {'关联度':<6} {'名称':<14} {'入选日期':<12} {'最近原因'}")
    for t in topics:
        reason = (t.get("ztnr") or "").replace("\n", " ")[:60]
        print(
            f"{t.get('id', ''):<8} {t.get('gld', ''):<6} {t.get('ztmc', ''):<14} "
            f"{t.get('rxsj', ''):<12} {reason}"
        )


def cmd_by_topic(args: argparse.Namespace) -> None:
    """按题材筛选代码,与 codes cache 做 join。"""
    tcache = TopicCache()
    cache = CodesCache()

    topic_id, topic_name = tcache.resolve_topic(args.topic, seed_code=args.seed, force=args.force)
    topic_payload = tcache.load_topic(args.seed, topic_id, force=args.force)

    codes_data = cache.ensure(force=args.force)
    codes_by_code = {r["code"]: r for r in codes_data["codes"]}

    rows = []
    for stock in topic_payload["stocks"]:
        code6 = stock.get("zqdm")  # 6 位代码
        code_info = codes_by_code.get(code6, {})
        rows.append({
            "rank": stock.get("pm", ""),
            "code6": code6,
            "full_code": code_info.get("full_code", ""),
            "name": stock.get("zqjc") or code_info.get("name", ""),
            "board": code_info.get("board", ""),
            "category": code_info.get("category", ""),
            "change_pct": stock.get("zdf"),
            "change_pct_5d": stock.get("zdf_5d"),
            "change_pct_20d": stock.get("zdf_20d"),
        })

    print(
        f"# topic={topic_name!r} (id={topic_id}), seed={args.seed}, "
        f"matched={len(rows)} / topic_size={topic_payload['count']}"
    )
    print(
        f"{'rank':<5} {'full_code':<12} {'name':<14} {'board':<22} "
        f"{'today%':>8} {'5d%':>8} {'20d%':>8}"
    )
    for r in rows[: args.limit]:
        today = f"{r['change_pct']:>7.2f}" if isinstance(r["change_pct"], (int, float)) else f"{'-':>7}"
        d5 = f"{r['change_pct_5d']:>7.2f}" if isinstance(r["change_pct_5d"], (int, float)) else f"{'-':>7}"
        d20 = f"{r['change_pct_20d']:>7.2f}" if isinstance(r["change_pct_20d"], (int, float)) else f"{'-':>7}"
        full = r["full_code"] or f"?{r['code6']}"
        print(
            f"{r['rank']:<5} {full:<12} {r['name']:<14} {r['board']:<22} "
            f"{today} {d5} {d20}"
        )
    if len(rows) > args.limit:
        print(f"... ({len(rows) - args.limit} more)")


def main() -> None:
    parser = argparse.ArgumentParser(description="eltdx 代码表本地缓存")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_refresh = sub.add_parser("refresh", help="强制从远端刷新")
    p_refresh.set_defaults(func=cmd_refresh)

    p_list = sub.add_parser("list", help="筛选并打印")
    p_list.add_argument("--board", help="按板块过滤, 如 sse_star_market")
    p_list.add_argument("--category", help="按品种过滤, 如 a_share/etf/index")
    p_list.add_argument("--exchange", help="按市场过滤, sh/sz/bj")
    p_list.add_argument("--name", help="按名称模糊匹配")
    p_list.add_argument("--limit", type=int, default=50)
    p_list.add_argument("--force", action="store_true")
    p_list.set_defaults(func=cmd_list)

    p_lookup = sub.add_parser("lookup", help="按代码精确查询")
    p_lookup.add_argument("code", help="完整代码 sh688825 或 6 位 688825")
    p_lookup.add_argument("--force", action="store_true")
    p_lookup.set_defaults(func=cmd_lookup)

    p_stats = sub.add_parser("stats", help="缓存统计与分组")
    p_stats.add_argument("--force", action="store_true")
    p_stats.set_defaults(func=cmd_stats)

    p_topics = sub.add_parser("topics", help="列出 seed_code 可访问的题材目录")
    p_topics.add_argument("--seed", default=DEFAULT_SEED_CODE, help=f"种子股票,默认 {DEFAULT_SEED_CODE}")
    p_topics.add_argument("--force", action="store_true")
    p_topics.set_defaults(func=cmd_topics)

    p_by_topic = sub.add_parser("by-topic", help="按题材筛选代码表")
    p_by_topic.add_argument("topic", help="题材名称或 ID, 支持模糊匹配")
    p_by_topic.add_argument("--seed", default=DEFAULT_SEED_CODE, help=f"种子股票,默认 {DEFAULT_SEED_CODE}")
    p_by_topic.add_argument("--limit", type=int, default=50)
    p_by_topic.add_argument("--force", action="store_true")
    p_by_topic.set_defaults(func=cmd_by_topic)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()