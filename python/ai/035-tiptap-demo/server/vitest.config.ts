import { fileURLToPath } from 'node:url'
import { defineConfig } from 'vitest/config'

/**
 * vitest 配置。
 *
 * 这个测试套件**很小**（18 个文件 / 264 个用例 / 全套 ~1s），
 * 但默认配置会让 vitest 按 CPU 核数开一堆 worker（这台 M4 上是 9 个 fork），
 * 每个 fork 都要：
 *   1. 重新把 vite 的模块图 transform 一遍（transform + collect ≈ 5.4s CPU）
 *   2. 重新 import 一遍 openai / ollama / @anthropic-ai/sdk（每次 ~180ms）
 *
 * 结果就是：跑一次测试 CPU 占用 ~500%、磁盘狂转 IO，其实 wall time 反而更慢
 * （并行拆分 + 重复 transform 的开销 > 省下的那点执行时间）。
 *
 * 这里做两件事把它压下去。
 */

function stub(name: string): string {
  return fileURLToPath(new URL(`./tests/stubs/${name}.ts`, import.meta.url))
}

export default defineConfig({
  resolve: {
    // 真 SDK 在测试里一律被 vi.mock 挡掉（见 tests/llmClients.test.ts），
    // 所以 alias 到零成本的桩，省掉每个测试文件 ~180ms 的重复 import。
    // 三个包必须指向三个不同文件 —— vitest 的 mock 注册表按解析后的路径做 key。
    alias: {
      ollama: stub('ollama'),
      openai: stub('openai'),
      '@anthropic-ai/sdk': stub('anthropic'),
    },
  },
  test: {
    // 套件太小，并行拆分带来的调度开销 > 收益：单 fork 反而更快。
    // 单 fork 下 CPU 从 ~500% 降到 ~110%，wall time 从 1.8s 降到 ~1s。
    pool: 'forks',
    poolOptions: {
      forks: {
        singleFork: true,
      },
    },
    // isolate 保持默认的 true —— 每个测试文件仍然拿到干净的模块表，
    // vi.mock / vi.spyOn 不会跨文件串味。
    isolate: true,
  },
})
