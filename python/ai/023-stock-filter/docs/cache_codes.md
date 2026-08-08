# cache_codes.py 使用文档

eltdx 全市场代码表(`SecurityCode`)的本地缓存与查询脚本。

## 一、用途

eltdx 协议 `0x044d` 在主站维护了一份覆盖沪深北三个市场的完整代码表,每个代码包含交易所、市场编号、名称、价格换算倍数、昨收、品种分类、所属板块等 14 个字段。`cache_codes.py` 把这份代码表**全量拉到本地**,再以 JSON 持久化,提供 4 类操作:

- `refresh` — 强制从远端拉取并落盘
- `list` — 按板块/品种/市场/名称筛选打印
- `lookup` — 按代码精确查询
- `stats` — 缓存统计与分组计数

## 二、安装与依赖

| 依赖 | 版本 | 说明 |
|---|---|---|
| Python | 3.11+ | 用了 `list[dict]`/`dict[str, int]` PEP 585 内建泛型语法 |
| eltdx | 1.3.0+ | 提供 `TdxClient` 与 `to_jsonable` |
| 网络 | TCP 7709 | 远端拉取走通达信主站 socket |

脚本只依赖项目已有的 `~/.pyenv/versions/qlib/bin/python`,无需额外 `pip install`。

## 三、缓存策略

| 配置 | 默认值 | 修改位置 |
|---|---|---|
| 缓存路径 | `<项目根>/.cache/eltdx_codes.json` | `CACHE_PATH` |
| TTL | 24 小时 (86400 秒) | `TTL_SECONDS` |
| 远端超时 | 5 秒/请求 | `TdxClient(timeout=5)` |
| 拉取市场 | sh / sz / bj 三市场全量 | `fetch_remote()` 内的 `for market in (...)` |

**触发刷新的时机**:
- 缓存文件不存在
- 缓存文件 mtime 距今 > TTL
- 显式调用 `refresh` 子命令或传 `--force`

其余所有读取都走本地 JSON,**不再产生网络请求**。实测二次查询耗时约 0.2 秒,首次拉取约 3 秒。

## 四、CLI 用法

### 4.1 通用语法

```
~/.pyenv/versions/qlib/bin/python test-scripts/cache_codes.py <子命令> [选项]
```

所有读操作 (`list / lookup / stats`) 都会先调 `CodesCache.ensure()`,自动判断 TTL 并按需 refresh。不想等自动刷新可以加 `--force` 立刻远端拉取。

### 4.2 refresh — 强制刷新

```
~/.pyenv/versions/qlib/bin/python test-scripts/cache_codes.py refresh
```

输出:
```
[cache] refreshing from remote...
[refresh] saved 51959 codes to .../.cache/eltdx_codes.json
```

适合场景:每天开盘前、刚装好环境、怀疑本地数据陈旧。

### 4.3 list — 筛选打印

```
~/.pyenv/versions/qlib/bin/python test-scripts/cache_codes.py list [选项]
```

| 选项 | 取值示例 | 说明 |
|---|---|---|
| `--board` | `sse_star_market` / `szse_main_board` / `szse_chinext` / `bse_listed_stock` | 按板块过滤 |
| `--category` | `a_share` / `b_share` / `etf` / `index` / `bond` | 按品种过滤 |
| `--exchange` | `sh` / `sz` / `bj` | 按交易所过滤 |
| `--name` | `长鑫` / `银行` | 名称模糊匹配(大小写不敏感) |
| `--limit` | 整数,默认 50 | 最多打印多少条 |
| `--force` | flag | 强制刷新缓存后再查 |

**示例**

```
# 科创板(616 只)
python cache_codes.py list --board sse_star_market

# 深圳主板的 A 股
python cache_codes.py list --exchange sz --board szse_main_board --category a_share

# 名称含 "银行" 的所有代码
python cache_codes.py list --name 银行

# 多条件组合:科创板 + A 股
python cache_codes.py list --board sse_star_market --category a_share
```

输出格式:

```
# total=51959, matched=616
full_code    name           board                  category   prev_close
sh688836     宇树科技           sse_star_market        a_share          0.00
sh688835     高凯技术           sse_star_market        a_share          0.00
...
```

### 4.4 lookup — 精确查询

```
python cache_codes.py lookup <代码>
```

支持完整代码 `sh688825` 或纯 6 位 `688825`。命中后打印该条记录的完整 JSON:

```
{
  "exchange": "sh",
  "market_id": 1,
  "code": "688825",
  "name": "长鑫科技",
  "previous_close_price": 51.96,
  "category": "a_share",
  "board": "sse_star_market",
  "full_code": "sh688825",
  ...
}
```

找不到时打印 `not found: <code>` 到 stderr 并以非零状态退出。

### 4.5 stats — 分组统计

```
python cache_codes.py stats
```

打印缓存元信息 + 按 exchange/board/category 的计数分布,用于快速摸清市场结构:

```
path        : .../.cache/eltdx_codes.json
fetched_at  : 2026-08-08 11:25:32
ttl_seconds : 86400
fresh       : True
total_codes : 51959

[by exchange]
  bj     368
  sh   27654
  sz   23937

[by board]
  sse_main_board            1699
  szse_main_board           1494
  szse_chinext              1403
  sse_star_market            616
  bse_listed_stock           336

[by category]
  a_share       5548
  index         1678
  etf           1578
  b_share         79
  bond            21
```

## 五、作为库引用

`cache_codes.py` 顶层暴露的 `CodesCache` 类可直接被其它脚本 import:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "test-scripts"))
from cache_codes import CodesCache

cache = CodesCache()
data = cache.ensure()  # 自动判断 TTL,过期则刷新

# 示例 1:拿所有科创板代码
star_codes = [c for c in data["codes"] if c["board"] == "sse_star_market"]

# 示例 2:用 set 做差集,找沪深 A 股
a_share = {
    c["full_code"] for c in data["codes"]
    if c["category"] == "a_share" and c["exchange"] in ("sh", "sz")
}

# 示例 3:按 prev_close_price 找高价股(>=100 元)
expensive = [c for c in data["codes"] if c["previous_close_price"] >= 100]
```

## 六、缓存文件格式

```
{
  "fetched_at": 1754628332.123,        # unix timestamp
  "fetched_at_iso": "2026-08-08 11:25:32",
  "count": 51959,
  "codes": [
    {
      "exchange": "sh",
      "market_id": 1,
      "code": "688825",
      "name": "长鑫科技",
      "multiple": 100,
      "decimal": 2,
      "previous_close_price": 51.96,
      "volume_ratio_base": 25655.74,
      "unknown0_raw": "796fc846",
      "previous_close_raw": "0ad74f42",
      "unknown3_raw": "2132e525",
      "category": "a_share",
      "category_reason": "SSE A-share code prefix",
      "board": "sse_star_market",
      "board_reason": "SSE STAR Market prefix",
      "full_code": "sh688825"
    },
    ...
  ]
}
```

**注意**:`unknown0_raw` / `previous_close_raw` / `unknown3_raw` 是底层协议字节,**不要参与业务逻辑**,仅供排查用。

## 七、板块/品种取值参考

### 7.1 board (板块)

| 取值 | 含义 |
|---|---|
| `sse_main_board` | 上交所主板 |
| `sse_star_market` | 上交所科创板 |
| `szse_main_board` | 深交所主板 |
| `szse_chinext` | 深交所创业板 |
| `bse_listed_stock` | 北交所 |
| `none` | 非股票品种(债券/基金等) |

### 7.2 category (品种)

| 取值 | 含义 |
|---|---|
| `a_share` | A 股 |
| `b_share` | B 股 |
| `etf` | ETF 基金 |
| `index` | 指数 |
| `bond` | 债券 |
| `private_convertible_bond` | 私募可转债 |
| `unknown` | 协议无法判定的品种(主要是其它) |

## 八、注意事项

1. **首次运行必须联网**:本地无缓存时所有读命令都会触发远端拉取
2. **25 MB JSON**:5.2 万条记录约占 25 MB 磁盘,如不在意可调低 TTL 以更激进刷新
3. **代码表变动稀少**:eltdx 协议每天只会更新少量新上市/退市代码,**24h TTL 已经足够**,不必高频刷新
4. **不要手工编辑 JSON**:`CodesCache.load()` 直接信任文件结构,改坏会导致下次读取异常
5. **真要订阅新上市代码**,需要把 `TTL_SECONDS` 调到几小时级,或者用 `--force` 主动刷新

## 九、扩展方向

| 需求 | 思路 |
|---|---|
| 按概念板块过滤 | 改用 `hot_topics` / `topic_stocks` helper,再做交集 |
| SQLite 后端 | 替换 `CodesCache` 持久化层为 `sqlite3`,支持 SQL |
| 增量刷新 | 对比 `fetched_at` 与上次 `count`,只拉新增/退市 |
| watch 模式 | `while True` 定时 refresh,差异行推到 stdout |