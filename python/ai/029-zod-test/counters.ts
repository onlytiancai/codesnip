// 进程级性能计数器：mitata 每个用例自己报 heap，这里再补一份全局 CPU / RSS / GC 汇总。
// 被 bench.ts 与 bench-myzod.ts 共用。

import { PerformanceObserver, constants } from "node:perf_hooks";

const GC_KIND: Record<number, string> = {
  [constants.NODE_PERFORMANCE_GC_MAJOR]: "major",
  [constants.NODE_PERFORMANCE_GC_MINOR]: "minor",
  [constants.NODE_PERFORMANCE_GC_INCREMENTAL]: "incremental",
  [constants.NODE_PERFORMANCE_GC_WEAKCB]: "weakcb",
};

export function startCounters() {
  const gcTally: Record<string, { count: number; pauseMs: number }> = {};
  let peakRss = process.memoryUsage().rss;

  const gcObserver = new PerformanceObserver((list) => {
    for (const entry of list.getEntries()) {
      const kind = GC_KIND[(entry.detail as any)?.kind] ?? "unknown";
      const slot = (gcTally[kind] ??= { count: 0, pauseMs: 0 });
      slot.count++;
      slot.pauseMs += entry.duration;
    }
  });
  gcObserver.observe({ entryTypes: ["gc"] });

  const rssPoller = setInterval(() => {
    const rss = process.memoryUsage().rss;
    if (rss > peakRss) peakRss = rss;
  }, 50);
  rssPoller.unref();

  const cpuBefore = process.cpuUsage();
  const memBefore = process.memoryUsage();
  const wallBefore = process.hrtime.bigint();

  return async function printCounters() {
    const wallMs = Number(process.hrtime.bigint() - wallBefore) / 1e6;
    const cpu = process.cpuUsage(cpuBefore);
    const memAfter = process.memoryUsage();
    if (memAfter.rss > peakRss) peakRss = memAfter.rss;
    // 基准跑的是同步循环，事件循环全程被占满，gc 观察者的回调此刻才有机会 flush
    await new Promise((r) => setTimeout(r, 100));
    gcObserver.disconnect();

    const mb = (n: number) => (n / 1024 / 1024).toFixed(2) + " MB";
    const total = Object.values(gcTally).reduce(
      (a, s) => ({ count: a.count + s.count, pauseMs: a.pauseMs + s.pauseMs }),
      { count: 0, pauseMs: 0 },
    );

    console.log("\n进程级汇总（整轮 benchmark，含 mitata 自身开销）");
    console.table({
      wall: { value: wallMs.toFixed(0) + " ms" },
      "cpu user": { value: (cpu.user / 1000).toFixed(0) + " ms" },
      "cpu system": { value: (cpu.system / 1000).toFixed(0) + " ms" },
      "cpu/wall": {
        value:
          (((cpu.user + cpu.system) / 1000 / wallMs) * 100).toFixed(1) + " %",
      },
      "rss 峰值": { value: mb(peakRss) },
      "rss Δ": { value: mb(memAfter.rss - memBefore.rss) },
      "heapUsed Δ": { value: mb(memAfter.heapUsed - memBefore.heapUsed) },
      "external Δ": { value: mb(memAfter.external - memBefore.external) },
      "gc 次数": { value: String(total.count) },
      "gc 暂停总计": { value: total.pauseMs.toFixed(1) + " ms" },
    });
    console.table(gcTally);
    console.log(
      `node ${process.version} · ${process.platform}/${process.arch} · zod ${
        (await import("zod/package.json", { with: { type: "json" } })).default
          .version
      }`,
    );
    console.log(
      "注：cpu/wall 统计所有线程（含 V8 GC 辅助线程），>100% 属正常；" +
        "内存 Δ 受 GC 时机影响可能为负；单用例的 heap 见上方表格。",
    );
  };
}
