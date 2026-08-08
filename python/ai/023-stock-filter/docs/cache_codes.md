# cache_codes.py 使用文档

eltdx 全市场代码表(`SecurityCode`)的本地缓存与查询脚本,附带按题材板块筛选的能力。

## 一、用途

eltdx 协议 `0x044d` 在主站维护了一份覆盖沪深北三个市场的完整代码表,每个代码包含交易所、市场编号、名称、价格换算倍数、昨收、品种分类、所属板块等 14 个字段。`cache_codes.py` 把这份代码表**全量拉到本地**,再以 JSON 持久化,提供 6 类操作:

- `refresh` — 强制从远端拉取并落盘
- `list` — 按板块/品种/市场/名称筛选打印
- `lookup` — 按代码精确查询
- `stats` — 缓存统计与分组计数
- `topics` — 列出某 seed 股票可访问的全部题材
- `by-topic` — 按题材名称或 ID 筛选代码,join 后输出行情摘要

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
| 代码表缓存路径 | `<项目根>/.cache/eltdx_codes.json` | `CACHE_PATH` |
| 代码表 TTL | 24 小时 (86400 秒) | `TTL_SECONDS` |
| 题材目录缓存路径 | `<项目根>/.cache/topics_seed.json` | `TOPICS_SEED_CACHE` |
| 题材成分股缓存目录 | `<项目根>/.cache/topics/` | `TOPIC_CACHE_DIR` |
| 题材缓存 TTL | 6 小时 (21600 秒) | `TOPIC_TTL_SECONDS` |
| 默认 seed 股票 | `sz000001` 平安银行 | `DEFAULT_SEED_CODE` |
| 远端超时 | 5 秒/请求 | `TdxClient(timeout=5)` |
| 拉取市场 | sh / sz / bj 三市场全量 | `fetch_remote()` 内的 `for market in (...)` |

**触发刷新的时机**:
- 代码表:缓存文件不存在 / mtime 超过 24h / `refresh` 子命令 / `--force`
- 题材:`by-topic` / `topics` 子命令首次拉取,或 6h TTL 到期,或 `--force`

题材 TTL 比代码表短,因为题材成分股**变动频繁**(新股纳入、老股退出、临时热点加入)。

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

### 4.6 topics — 列出 seed 股票可访问的题材目录

```
python cache_codes.py topics [选项]
```

题材(概念板块)在 eltdx 中以 `seed_code + topic_id` 索引:每个题材都属于某些"种子"股票,其它股票通过 `topic_compare(seed_code, topic_id)` 反查。默认 seed 是 `sz000001` 平安银行,但**不同 seed 看到的题材集合不同**,深度个股(题材丰富)的 seed 更全面。

| 选项 | 取值示例 | 说明 |
|---|---|---|
| `--seed` | `sh688825` (默认 `sz000001`) | 种子股票 |
| `--force` | flag | 强制重新拉取 seed 目录 |

**示例**

```
# 默认 seed,题材少但权威
python cache_codes.py topics

# 用长鑫科技做 seed,题材丰富(16 个)
python cache_codes.py topics --seed sh688825
```

输出格式:

```
# seed=sh688825, total=16
id       关联度    名称             入选日期         最近原因
626      5      次新股            20260727     公司于2026-07-27在上交所科创板上市
31       5      芯片             20260727     公司已成长为中国第一、全球第四的DRAM厂商...
2945     3      存储芯片           20260727     公司的主营业务为DRAM产品的研发、设计、生产及销售...
...
```

### 4.7 by-topic — 按题材筛选代码

```
python cache_codes.py by-topic <题材名称或 ID> [选项]
```

从 seed 股票的题材目录里**模糊匹配**名称(或精确匹配 ID),拿到 `topic_id` 后调用 `topic_compare` 拉取成分股,再与 codes cache 做 join,输出 8 列行情摘要。

| 选项 | 取值示例 | 说明 |
|---|---|---|
| `topic` (位置参数) | `存储芯片` / `芯片` / `31` / `1087` | 题材名(模糊匹配)或 ID |
| `--seed` | `sh688825` (默认 `sz000001`) | 种子股票 |
| `--limit` | 整数,默认 50 | 最多打印多少条 |
| `--force` | flag | 强制重拉题材快照 |

**匹配优先级**:
1. 按 `id` 精确匹配(整数或字符串)
2. 按 `ztmc` 名称精确匹配
3. 按 `ztmc` 名称包含匹配,**若多个结果则列出候选并退出**(避免歧义)
4. 无匹配 → 报错退出

**示例**

```
# 按名称精确匹配
python cache_codes.py by-topic 存储芯片 --seed sh688825

# 按 ID
python cache_codes.py by-topic 31 --seed sh688825

# 按名称模糊匹配(唯一命中)
python cache_codes.py by-topic 芯片 --seed sh688825

# 模糊匹配多个,会列出候选:
python cache_codes.py by-topic 概念 --seed sh688825
[topics] ambiguous match for '概念', candidates:
  - id=2554   name=消费电子概念
  - id=2317   name=抖音概念
  ...
```

输出格式:

```
# topic='存储芯片' (id=2945), seed=sh688825, matched=20 / topic_size=20
rank  full_code    name           board                    today%      5d%     20d%
1     sh688419     耐科装备           sse_star_market          20.00   46.95  -27.36
2     sh688519     南亚新材           sse_star_market          16.53   38.78   -8.87
3     sz301511     德福科技           szse_chinext             16.21   37.10  -21.99
...
```

**字段说明**:
- `rank` — 题材内排名(pm,按涨跌幅排序)
- `full_code` — 完整代码,从 codes cache join 出来
- `name` — 优先取 F10 简称,缺失时退到 codes cache
- `board` — 板块(从 codes cache),用于筛选"同题材 + 同板块"等组合
- `today%` / `5d%` / `20d%` — F10 给的当日 / 5 日 / 20 日涨跌幅

### 4.8 理解 Seed(题材查询的入口)

题材(`topic`)在 eltdx 中**没有公开的全量目录 API**,查询方式是"以股票为锚点反查":

```
题材 = (seed_code, topic_id, ...)
       ↑         ↑
    哪只股票    这只股关联的某个题材
```

也就是说,**先选一只股票作为"种子",再问"它关联了哪些题材"**。换一只 seed,看到的题材集合就不同。这是 eltdx 协议本身的设计,不是脚本的限制。

**常用 seed 推荐**

| 场景 | 推荐 seed | 题材数量 | 典型题材 |
|---|---|---|---|
| 金融/银行 | `sz000001` 平安银行 | ~10 | 大盘股、高股息、保险重仓、低市净率 |
| 半导体/存储/芯片 | `sh688825` 长鑫科技 | ~16 | 存储芯片、芯片、大基金持股、次新股 |
| 消费/白酒 | `sh600519` 贵州茅台 | ~12 | 茅台概念、消费、大盘股 |
| 题材广度 | 任选行业龙头,多个 seed 互补 | 各异 | — |
| 查具体题材 | **必须是该题材内的股票** | — | 否则 `topic_compare` 返回空 |

**发现合适 seed 的 3 种方法**

1. **已知题材名**:先挑一只该题材内的知名股做 seed,跑 `topics --seed` 看它有哪些题材
2. **一次性拿全题材 ID**:用 `client.f10.topic_ids(code)` 在 Python 里直接拿
3. **题材 ID 是稳定主键**:同一个题材(例如"存储芯片" id=2945),不管用哪只关联股票做 seed,`topic_compare` 返回的成分股一致

**当前脚本的默认行为**

```python
DEFAULT_SEED_CODE = "sz000001"  # 平安银行
```

平安银行题材少但**权威**,适合基础演示。**做题材筛选务必显式指定题材丰富的 seed**:

```bash
python cache_codes.py topics --seed sh688825
python cache_codes.py by-topic 存储芯片 --seed sh688825
```

**已知限制**

`by-topic` 当前**只从一个 seed 看题材**,看不到该 seed 没关联的题材。要凑齐全集,需要遍历 3-5 个题材互不重叠的 seed(银行 / 科技 / 消费 / 医药 / 周期)并 union 它们各自的题材目录 — 这是协议层限制,扩展方向见 [九、扩展方向](#九扩展方向)。

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
| 全市场题材筛选 | 多 seed union:遍历 3-5 个题材互补的 seed(银行/科技/消费/医药/周期),合并题材目录,再去重成分股 |
| SQLite 后端 | 替换 `CodesCache` 持久化层为 `sqlite3`,支持 SQL |
| 增量刷新 | 对比 `fetched_at` 与上次 `count`,只拉新增/退市 |
| watch 模式 | `while True` 定时 refresh,差异行推到 stdout |
| 与短线指标 join | 在 `by-topic` 输出基础上追加 `eltdx_shortline_indicators` 的字段 |

## 十、题材缓存结构

`.cache/topics_seed.json`(单文件):

```json
{
  "seed_code": "sh688825",
  "fetched_at": 1754628332.12,
  "fetched_at_iso": "2026-08-08 11:25:32",
  "topics": [
    {"id": 626, "ztmc": "次新股", "gld": 5, "rxsj": 20260727, "ztnr": "...", ...},
    ...
  ]
}
```

`.cache/topics/<seed>__<topic_id>.json`(每个题材一份):

```json
{
  "seed_code": "sh688825",
  "topic_id": "2945",
  "fetched_at": 1754628400.34,
  "fetched_at_iso": "2026-08-08 11:26:40",
  "count": 20,
  "stocks": [
    {"pm": 1, "zqdm": "688419", "zqjc": "耐科装备", "zdf": 20.0, "zdf_5d": 46.95, "zdf_20d": -27.36, ...},
    ...
  ]
}
```

**注意**:`zqdm` 是 6 位裸代码(无市场前缀),与 codes cache 中的 `code` 字段同义;join 时用 `code == zqdm` 而不是 `full_code == zqdm`。