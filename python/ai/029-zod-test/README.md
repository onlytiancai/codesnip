# zod 性能测试 + myzod

用 [mitata](https://github.com/evanwashere/mitata) 对 zod 4.5 的 `parse` / `safeParse` 做性能基准（时间、CPU、内存、GC），
并实现一个 codegen 版的最小替代品 **myzod** 做对比。

环境：Node 24（原生执行 `.ts`）· Apple M4 · zod 4.5.4 · mitata 1.0.34

```bash
pnpm install
pnpm bench          # zod 各场景基准
pnpm bench:myzod    # myzod vs zod 对比
```

两个脚本都支持 `--json`（机器可读，便于版本间对比）和 `--filter=<正则>`（按 **bench 名**过滤，不是 group 名）。

## 文件

| 文件 | 作用 |
|---|---|
| `zod-test.ts` | 最小示例，保持不动 |
| `bench.ts` | zod 基准：simple / nested / array / union / schema 构建 |
| `myzod.ts` | codegen 版校验器，只实现 `safeParse` |
| `bench-myzod.ts` | myzod vs zod 对比，**含正确性断言** |
| `counters.ts` | 进程级 CPU / RSS / GC 计数器，两个 bench 共用 |

## 结果一：zod 自身

| 场景 | 耗时 | 相对手写校验 |
|---|---|---|
| 简单对象 `safeParse` valid | 125 ns | 5x 慢 |
| 简单对象 `safeParse` invalid | 1.37 µs | — |
| `parse` invalid + try/catch | 3.40 µs | 比 safeParse invalid 再慢 2.5x |
| 100 元素数组 | 19.3 µs | 7x 慢 |
| `discriminatedUnion` vs `union` | 33 ns vs 131 ns | **4x** |
| 复用 schema vs 每次新建 | 181 ns vs 24 µs | **133x** |

**结论**：schema 必须缓存复用；能用 `discriminatedUnion` 就别用 `union`；
错误路径比成功路径贵一个数量级，抛异常比 `safeParse` 再贵 2.5 倍。

## 结果二：myzod vs zod（同一份 Nested schema）

| 场景 | zod | myzod | 倍数 |
|---|---|---|---|
| valid | 314 ns / 1.91 kb | **65 ns / 336 b** | **5.2x** |
| sparse + 未知字段 | 293 ns / 1.62 kb | **63 ns / 224 b** | **5.4x** |
| invalid | 2.27 µs / 7.73 kb | 7 ns / 80 b | 332x ⚠️ |
| 构建 + 首次 parse | 83 µs | **10.6 µs** | 7.8x |

稳态路径快 **5x**、堆分配降到 **1/6**，这是可信的核心数字。

> ⚠️ **invalid 那个 332x 不要当真。** myzod 是 fail-fast 且测试数据第一个字段 `id` 就错了，
> 几乎立刻返回；zod 则遍历全部字段收集完整 issues。这是语义差异，不是实现优劣。
> 保留它是为了不藏东西，group 名称上已标注「非等价」。

## myzod 是怎么快的

不是靠少校验，而是 **codegen**：schema 树在首次 `safeParse` 时经 `new Function`
编译成一个完全内联的扁平函数并缓存。zod 慢在每个字段都要走一层通用的对象/闭包分发，
而编译后整棵 `Nested` 展开成约 40 行顺序判断——无函数调用、无递归、字段访问单形态。
正则和 default 值经闭包数组 `C` 传入，不序列化进源码；错误 `path` 在编译期就是常量字符串。

```js
S.source()   // 查看生成的源码
```

几个实现要点：

- **object**：`typeof/null/Array` 三连判断后逐字段取值，最后用对象字面量**一次性构造**——
  单形态，且天然实现了 strip 未知字段（比 `{...input}` 再 `delete` 快得多）
- **optional 字段**：生成 2ⁿ 个完整字面量分支（`geo` 有/无各一个），避免「先建对象再挂属性」
  引起隐藏类迁移。optional 超过 2 个时退回「先建再挂」，否则 2ⁿ 爆炸
- **array**：`new Array(len)` 预分配 + 索引循环，不用 `map`
- **enum**：3 个成员直接展开成 `v==="admin"||v==="user"||v==="guest"`，比 `Set.has` 快

### 支持的类型

`string`（`.min` `.length`）· `email` · `number`（`.int` `.min`）· `object` · `array` · `enum` ·
`.optional()` · `.default()`。**刚好覆盖 `Nested`，没有更多**——没有 union / refine / transform / 异步。

### 语义取舍

1. **剔除未知字段**，与 zod 的 strip 对齐（保证对比公平）
2. **fail-fast**，只报第一个错误 `{ path, message }`，与 zod 收集全部 issues 不同
3. **不深拷贝**：只重建 schema 声明到的层级，叶子值直接引用
4. `.default([])` 与 zod 一致地**共享同一个实例**——改动返回的 `tags` 会污染后续所有结果

正确性由 13 组断言在 benchmark **之前**卡住（valid / sparse / invalid / zip 长度 5-6 /
age 17-18 / age 非整数 / role 非法 / tags 含非字符串 / 顶层 null、数组、字符串），
逐一与 zod 的 `safeParse` 结果 `deepStrictEqual`，任一不一致直接抛错退出。
一个跑得快但结果不对的校验器没有意义。

## 踩过的坑

### 1. `gc('inner')` 会把快用例放大 43 倍

最初给所有用例加了 `.gc("inner")`，结果同一个用例在全量跑时报 36 ns、单独跑时报 2.3 µs。
`gc('inner')` 每个采样都强制一次 full GC，把约 1 ms 的暂停折进了计时。

改用 **`.gc("once")`**：计时准确（40 ns），且**仍保留 heap/iter 统计**，
只损失 mitata 的 per-sample GC 暂停列——那个由 `counters.ts` 的进程级统计补上。

### 2. GC 计数器读数为 0

`PerformanceObserver` 的回调在 `disconnect()` 前没来得及 flush——benchmark 全程阻塞事件循环。
`counters.ts` 里先 `await` 100 ms 再 disconnect。同一原因让 RSS 轮询被饿死，
导致「峰值」比「增量」还小，现在会与最终读数取 max 兜底。

### 3. 读数注意

- `cpu/wall` 常显示 120–140%，因为 `process.cpuUsage()` 统计所有线程（含 V8 GC 辅助线程），
  **不是**单核利用率
- 内存 Δ 受 GC 时机影响可能为负，属正常
- `schema 构建` 用例的 max 会出现 4 ms 级离群值，那是一次 major GC，不是真实尾延迟
