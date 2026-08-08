"""eltdx 代码表本地缓存。

策略:
- 全量拉取 sh/sz/bj 三个市场的 SecurityCode
- 以 JSON 持久化(eltdx 的 SecurityCode 自带 to_jsonable)
- 默认 TTL = 24h;超过则下次调用时强制刷新
- 提供按板块/品种筛选的派生视图

用法:
    ~/.pyenv/versions/qlib/bin/python test-scripts/cache_codes.py refresh
    ~/.pyenv/versions/qlib/bin/python test-scripts/cache_codes.py list --board sse_star_market
    ~/.pyenv/versions/qlib/bin/python test-scripts/cache_codes.py list --category a_share
    ~/.pyenv/versions/qlib/bin/python test-scripts/cache_codes.py lookup sh688825
    ~/.pyenv/versions/qlib/bin/python test-scripts/cache_codes.py stats
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
TTL_SECONDS = 24 * 3600


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

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()