`GET /_nodes/hot_threads` 是定位 Elasticsearch 高 CPU 问题的核心工具。它会返回节点上最“热”（即消耗最多 CPU 时间）的线程堆栈，帮助你快速定位引发问题的具体代码段。

### 🚀 最简启动命令

当CPU飙升时，可以快速执行以下命令，捕获更全面的线程信息用于分析：

```bash
# 获取前10个最"热"的线程，采样间隔500ms，快照20次
curl -X GET "localhost:9200/_nodes/hot_threads?threads=10&interval=500ms&snapshots=20"
```

### ⚙️ 核心参数解析
`hot_threads` API 通过参数调节采样范围与细节，默认仅返回3个热点线程，建议诊断时调大参数以获得更多信息。

| 参数 | 默认值 | 诊断建议 | 作用 |
| :--- | :--- | :--- | :--- |
| `type` | `cpu` | `cpu` (推荐) | 采样类型，`cpu`=采样CPU时间；`block`=采样线程阻塞状态。 |
| `threads` | `3` | `10` 或 `9999` | 返回的热门线程数，诊断时**建议调高以发现更多细节**。 |
| `interval` | `500ms` | `1s` 或 `5s` | 两次采样的间隔，较长的间隔有助于识别持续性问题。 |
| `snapshots` | `10` | `20` 或 `30` | 采样的堆栈快照数量，更多快照有助于区分瞬发尖刺和持续消耗。 |

### 🔍 快速诊断步骤

1.  **确认CPU高的节点**：使用 `GET /_cat/nodes?v&s=cpu:desc`，查看 `cpu` 列数值最高的节点。
2.  **获取热点线程**：针对高CPU节点，请求 `GET /_nodes/<node_id>/hot_threads?threads=10`。
3.  **检查正在运行的任务**：调用 `GET /_tasks?detailed=true&actions=*search` 查看是否有长时间运行的搜索任务。

### 🎯 解读输出：三种搜索相关的高CPU模式
`hot_threads` 返回纯文本，核心部分是从线程堆栈中分离出的方法调用。以下是三种常见的**搜索场景**相关高CPU模式及其定位方法。

#### 模式一：低效查询 (`search` 线程)
线程名可能为 `[search][T#xx]`，堆栈显示某条具体查询消耗了大量CPU。

**1. 识别症状**：堆栈中包含 `lucene` 包的 `matchQuery`、`TermQuery`、`regexQuery` 等解析或执行方法，或线程名中包含 `search`。
**2. 问题确认**：这类线程表明当前执行的搜索查询计算成本高，可能包含低效的查询类型（如通配符 `*term*`、正则、脚本查询）或存在大聚合。
**3. 解决方案**：
*   使用 `GET _tasks?detailed=true&actions=*search` 找到对应的任务 ID。
*   若需紧急止血，使用 `POST _tasks/<task_id>/_cancel` 取消该任务。
*   为查询设置 `timeout` 参数。
*   分析并优化该查询（使用 `Profile` API），避免低效的查询模式。

#### 模式二：大量小数据段合并 (`merge` 线程)
线程名可能为 `[merge][T#xx]`，堆栈显示与 `IndexWriter` 和 `SegmentMerger` 相关。

**1. 识别症状**：`merge` 线程频繁出现且CPU占用高。
**2. 问题确认**：表明 Lucene 底层在进行大量的小段合并操作，通常与索引写入模式有关。
**3. 解决方案**：
*   优化索引写入策略，如增加 `refresh_interval` 或调整 `bulk` 请求大小。
*   考虑在业务低峰期执行 `force-merge`。

#### 模式三：高频刷新 (`refresh` 线程)
线程名可能为 `[refresh][T#xx]`，堆栈中包含 `IndexWriter` 的 `doFlush` 或 `refresh` 相关方法。

**1. 识别症状**：`refresh` 线程占用高。
**2. 问题确认**：表明索引的 `refresh_interval` 设置过短，导致频繁将内存中的数据刷新到磁盘，消耗大量 CPU。
**3. 解决方案**：
*   适当调大索引的 `refresh_interval` 设置。

### 💡 主动诊断脚本
为了进行更深入的诊断，可以编写一个简单的 bash 脚本，在高 CPU 期间自动收集多轮数据：

```bash
#!/bin/bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="hot_threads_diagnostic_${TIMESTAMP}"
mkdir -p $OUTPUT_DIR
for i in {1..5}; do
  echo "Capturing hot threads, iteration $i"
  curl -s "localhost:9200/_nodes/hot_threads?threads=20&interval=1s&snapshots=30" > "${OUTPUT_DIR}/hot_threads_${i}.txt"
  sleep 5
done
echo "Diagnostic data saved in $OUTPUT_DIR"
```

通过对比不同时间点的堆栈信息，可以更清晰地判断 CPU 占用是瞬时的还是持续的。

### ✨ 优化措施
根据 `hot_threads` 定位的根因，可采取不同优化方案：

| 根因类型 | 优化方案 |
| :--- | :--- |
| **低效查询** | 开启慢查询日志，优化查询语句（避免模糊、通配符），使用过滤器（filter）代替查询（query） |
| **负载过高** | 增加集群节点数量，升级硬件配置 |
| **批量请求过大** | 减小 `bulk` 请求的大小，降低单次搜索的并发量 |
| **配置不合理** | 增加节点CPU核数，调整 `search` 线程池大小，设置合理的 `refresh_interval`，如 `PUT /index/_settings { "index.refresh_interval": "30s" }` |

### 🛠️ 相关API对比
定位问题时可组合多种 API：

| API | 作用 | 使用场景 |
| :--- | :--- | :--- |
| `hot_threads` | 查看 CPU 消耗的代码栈 | 定位具体是哪段代码（如哪个查询）消耗了 CPU |
| `_tasks` | 查看集群中正在执行的任务 | 找出当前正在运行的长耗时任务（如搜索、写入） |
| `_cat/thread_pool` | 查看各线程池的状态 | 判断是否有线程池队列堆积（`queue` 数值高）或请求被拒绝（`rejected` 数值高） |

### 💎 总结
`hot_threads` API 的精髓在于，它将宏观的系统现象（“CPU高了”）微观化到了具体的代码执行路径。掌握这个工具，就等于拥有了在 CPU 飙高时进行高效“听诊”的能力。

你目前遇到高 CPU 问题时，主要是搜索缓慢、写入被拒，还是节点无响应？可以跟我说说具体现象，我再结合你的场景帮你看看排查思路。