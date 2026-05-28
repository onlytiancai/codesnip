## 用 `_nodes/hot_threads` 定位搜索高 CPU

这个 API 是排查 ES 高 CPU 的利器，核心思路是**抓实时热点线程堆栈**，看哪段代码在吃 CPU。

### 基本用法

```bash
GET _nodes/hot_threads
```

几个常用参数：

| 参数 | 说明 |
|------|------|
| `interval` | 采样间隔，默认 `500ms` |
| `snapshots` | 采样次数，默认 `10` |
| `threads` | 展示多少条热点线程，默认 `3` |
| `type` | `cpu`（默认）/ `wait` / `block` — 看你怀疑哪种 |

比如你怀疑搜索问题，用 CPU 模式多采几次：

```bash
GET _nodes/hot_threads?interval=1s&snapshots=5&type=cpu
```

### 怎么看结果

返回会按节点分组，每个节点列出热点线程，典型格式：

```
57.2% | node_1 | thread_name | [index_name] | time_spent
```

重点盯两类：

1. **`search` 相关线程**（`transport_search_action` / `search-worker`）
   - 说明有查询正在执行，看它在搜哪个索引/分片
   - 常见原因：深度分页、scroll 查询、`query_string` 复杂查询、大数据量 `terms` 聚合

2. **`coordinate_search`**（协调节点）
   - 多个 shard 查完了在合并结果，说明查询本身扫描了大量数据

### 配合定位的具体思路

**第一步：确认是哪个节点**
```bash
GET _nodes/hot_threads?nodes=node_id
```

**第二步：找到对应索引**
看返回里有没有 `[index_name]`，或者结合：
```bash
GET _cat/indices?v&h=health,index,docs.count,query_count,search.time_current
```

**第三步：找具体慢查询**
```bash
GET _searchslow?index=your_index   # 旧版本
# 新版本用：
GET /<index>/_search?explain=true   # 看执行计划
```

或者直接查慢日志配置：
```bash
GET _cluster/settings?include_defaults=true | grep "index.search.slowlog"
```

### 常见导致高 CPU 的搜索原因

| 场景 | 表现 | 解决方向 |
|------|------|---------|
| 深度分页 / scroll | `from + size` 很大 | 改用 `search_after` |
| `query_string` | 扫描全字段 | 换 `match` / `term` |
| 大 `terms` 聚合 | 聚合基数太大 | 调 `size`、用 `composite` 分页 |
| 正则查询 | 全索引扫描 | 用 `match` 代替 |
| join 类型查询 | `has_child` / `nested` | 考虑数据模型去关联 |

---

你现在是发现了高 CPU 在查 hot_threads，还是想了解排查的完整链路？如果有具体的返回内容贴给我，我帮你看看是哪个方向的问题。