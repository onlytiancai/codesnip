当 Elasticsearch 因“搜索请求”导致 CPU 飙高时，`GET _nodes/hot_threads` 是最直接的定位工具之一。关键不是只看“谁 CPU 高”，而是看 **哪些线程在忙、忙什么、调用栈停在哪里**。

下面给你一个实战化排查流程。

---

## 1. 先确认是不是搜索导致的 CPU

先看节点整体 CPU：

```bash
GET _cat/nodes?v&s=cpu:desc
```

重点字段：

* `cpu`：当前 CPU 使用率
* `node.role`
* `heap.percent`
* `ram.percent`

如果某几个 data node CPU 明显高，再去抓这些节点的 hot threads。

也建议同时看搜索线程池：

```bash
GET _cat/thread_pool/search?v&h=node_name,active,queue,rejected,completed
```

如果：

* `active` 很高
* `queue` 持续增长
* `rejected` 出现

通常就是搜索压力导致。

---

## 2. 正确抓取 hot threads

默认参数经常不够用。

推荐：

```bash
GET _nodes/hot_threads?threads=9999&type=cpu&interval=500ms&snapshots=10
```

参数解释：

* `threads=9999`
  → 不只看 top3，避免漏掉问题线程

* `type=cpu`
  → 查 CPU 问题（还有 wait/block）

* `interval=500ms`
  → 采样间隔

* `snapshots=10`
  → 多次采样避免偶然性

高 CPU 时，**连续抓 3~5 次** 很重要：

```bash
GET _nodes/hot_threads?threads=9999&type=cpu
```

间隔几秒抓一次。

如果总是同样线程，基本就是元凶。

---

## 3. 看输出时最重要的是线程名

输出长这样：

```text
::: node-01
Hot threads at 2026-05-28T10:00:00Z

97.2% cpu usage by thread
'elasticsearch[node-01][search][T#5]'
```

这里：

```text
[search][T#5]
```

就是搜索线程池。

常见线程池：

| 线程                      | 含义            |
| ----------------------- | ------------- |
| `[search]`              | 查询执行          |
| `[search_coordination]` | 协调节点聚合结果      |
| `[write]`               | 写入            |
| `[refresh]`             | refresh       |
| `[merge]`               | segment merge |

如果是：

```text
[search]
```

基本确定是搜索请求导致 CPU。

---

## 4. 真正关键：看调用栈（stack trace）

只看线程名不够。

要看下面停在哪里。

### 场景 A：全文检索 / Query 太复杂

如果看到：

```text
org.apache.lucene.search.BooleanWeight
org.apache.lucene.search.BooleanScorer
org.apache.lucene.search.Weight$DefaultBulkScorer.score
```

说明：

> Query 本身计算量很大

常见原因：

* 超大 bool query
* should 太多
* wildcard
* regexp
* fuzzy query
* nested query

例如：

```json
{
  "query": {
    "bool": {
      "should": [
        ...
        1000 clauses
      ]
    }
  }
}
```

这种非常容易打爆 CPU。

---

### 场景 B：脚本导致 CPU 高

如果出现：

```text
org.elasticsearch.script
org.elasticsearch.painless
PainlessScript
```

例如：

```text
org.elasticsearch.painless.PainlessScript$Script.execute
```

说明：

> script query / script score / runtime field 在烧 CPU

典型问题：

```json
"script_score": {
  "script": {
    "source": "cosineSimilarity(...)"
  }
}
```

或者：

```json
"runtime_mappings"
```

优化方向：

* 改成预计算字段
* 减少 script_score
* 减少 runtime field

---

### 场景 C：聚合（Aggregation）炸 CPU

如果看到：

```text
org.elasticsearch.search.aggregations
TermsAggregator
GlobalOrdinalsStringTermsAggregator
BucketsAggregator
```

说明：

> aggregation 太重

尤其：

```text
GlobalOrdinalsStringTermsAggregator
```

一般是：

```json
terms aggregation
```

在高基数字段上跑。

例如：

```json
{
  "aggs": {
    "uid": {
      "terms": {
        "field": "user_id"
      }
    }
  }
}
```

`user_id` 几千万 distinct values。

CPU 很容易爆。

优化：

* 降低 `size`
* 用 `composite aggregation`
* 避免高基数字段 terms agg
* 开 eager_global_ordinals

---

### 场景 D：排序导致 CPU

如果调用栈：

```text
TopFieldCollector
FieldComparator
```

说明：

> sort 很重

典型：

```json
"sort": [
  {
    "price": "asc"
  }
]
```

但字段没有 doc_values / 排序代价高。

或者：

```json
script sort
```

尤其危险。

---

### 场景 E：Fetch phase 太重

如果看到：

```text
FetchPhase
StoredFieldsPhase
SourceLookup
```

说明：

> 查出来后取 `_source` 很贵

常见：

* `_source` 很大
* 一次查很多文档
* highlight
* nested source

比如：

```json
"size": 10000
```

容易 CPU 高。

---

### 场景 F：协调节点 CPU 高

线程：

```text
[search_coordination]
```

调用栈：

```text
SearchPhaseController
QueryPhaseResultConsumer
```

说明：

> coordinator node 在 merge shard result

通常是：

* shard 太多
* fan-out 太大
* 一次查询 index 太多

比如：

```text
log-* (几千 shards)
```

优化：

* 减 shard 数
* routing
* 缩小查询范围
* rollover

---

## 5. hot_threads 和 task API 联动

只看 hot threads 有时不知道是谁发的 query。

同时查：

```bash
GET _tasks?actions=*search&detailed
```

找长时间运行的搜索：

```json
{
  "running_time_in_nanos": ...
}
```

重点看：

* source
* description
* parent_task_id

能看到真实 query。

有时直接定位：

> 某个 dashboard 的 DSL 把 CPU 打满。

---

## 6. 慢日志一起开（非常有效）

打开 search slowlog：

```json
PUT index_name/_settings
{
  "index.search.slowlog.threshold.query.warn": "1s",
  "index.search.slowlog.threshold.fetch.warn": "500ms",
  "index.search.slowlog.level": "info"
}
```

然后结合：

* hot_threads
* tasks
* slowlog

三板斧。

基本都能定位。

---

## 7. 一个真实排查思路

假设 hot_threads：

```text
93% cpu by thread
[search][T#12]

org.apache.lucene.search.BooleanScorer.score
org.elasticsearch.search.query.QueryPhase.execute
```

推断：

> query phase 很重。

再查：

```bash
GET _tasks?actions=*search&detailed
```

发现：

```json
wildcard: "*iphone*"
```

且扫：

```text
100+ indices
3000 shards
```

结论：

> wildcard + 大范围 shard fanout 导致 CPU 飙升。

不是 JVM 问题，也不是 GC。

---

### 一句话经验

看 `hot_threads` 时优先级：

**线程池名字 → 调用栈停在哪里 → 连续多次是否一致 → task API 找真实 query → slowlog 复现**

如果你贴一段 `GET _nodes/hot_threads` 输出（几十行即可），我可以直接帮你判断是哪类搜索在烧 CPU。
