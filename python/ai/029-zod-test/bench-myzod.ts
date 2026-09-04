import { run, bench, group, summary, do_not_optimize } from "mitata";
import { deepStrictEqual, strictEqual } from "node:assert";
import { z } from "zod";
import { m } from "./myzod.ts";
import { startCounters } from "./counters.ts";

// ---------------------------------------------------------------------------
// 同一份 Nested schema 的两种写法
// ---------------------------------------------------------------------------

const ZodNested = z.object({
  id: z.string(),
  profile: z.object({
    user: z.object({
      name: z.string().min(1),
      email: z.email(),
      age: z.number().int().min(18),
    }),
    address: z.object({
      city: z.string(),
      zip: z.string().length(6),
      geo: z.object({ lat: z.number(), lng: z.number() }).optional(),
    }),
  }),
  tags: z.array(z.string()).default([]),
  role: z.enum(["admin", "user", "guest"]).default("user"),
});

const makeMyNested = () =>
  m.object({
    id: m.string(),
    profile: m.object({
      user: m.object({
        name: m.string().min(1),
        email: m.email(),
        age: m.number().int().min(18),
      }),
      address: m.object({
        city: m.string(),
        zip: m.string().length(6),
        geo: m.object({ lat: m.number(), lng: m.number() }).optional(),
      }),
    }),
    tags: m.array(m.string()).default([]),
    role: m.enum(["admin", "user", "guest"]).default("user"),
  });

const MyNested = makeMyNested();

// ---------------------------------------------------------------------------
// 数据集
// ---------------------------------------------------------------------------

const validUser = { name: "Tom", email: "tom@example.com", age: 20 };

const valid = {
  id: "u-1",
  profile: {
    user: validUser,
    address: { city: "Shanghai", zip: "200000", geo: { lat: 31.2, lng: 121.5 } },
  },
  tags: ["a", "b"],
  role: "admin",
};

const invalid = {
  id: 1,
  profile: {
    user: { name: "", email: "not-an-email", age: 12 },
    address: { city: "Shanghai", zip: "20" },
  },
};

// 缺 geo / 缺 tags / 缺 role，且带一堆未声明的字段 —— 同时考察 default 填充与 strip
const sparseWithExtras = {
  id: "u-2",
  profile: {
    user: { ...validUser, nickname: "T" },
    address: { city: "Beijing", zip: "100000", country: "CN" },
    extra: 1,
  },
  createdAt: "2026-01-01",
};

// ---------------------------------------------------------------------------
// 正确性断言 —— 跑得快但结果不对的校验器没有意义，先卡住再 benchmark
// ---------------------------------------------------------------------------

function assertSame(label: string, input: unknown) {
  const zr = ZodNested.safeParse(input);
  const mr = MyNested.safeParse(input);
  strictEqual(mr.success, zr.success, `${label}: success 不一致`);
  if (zr.success && mr.success)
    deepStrictEqual(mr.data, zr.data, `${label}: data 不一致`);
}

for (const [label, input] of [
  ["valid", valid],
  ["sparse + 未知字段", sparseWithExtras],
  ["invalid", invalid],
  ["zip 长度 5", { ...valid, profile: { ...valid.profile, address: { city: "X", zip: "12345" } } }],
  ["zip 长度 6", { ...valid, profile: { ...valid.profile, address: { city: "X", zip: "123456" } } }],
  ["age 17", { ...valid, profile: { ...valid.profile, user: { ...validUser, age: 17 } } }],
  ["age 18", { ...valid, profile: { ...valid.profile, user: { ...validUser, age: 18 } } }],
  ["age 非整数", { ...valid, profile: { ...valid.profile, user: { ...validUser, age: 20.5 } } }],
  ["role 非法", { ...valid, role: "root" }],
  ["tags 含非字符串", { ...valid, tags: ["a", 1] }],
  ["顶层非对象", "nope"],
  ["顶层 null", null],
  ["顶层数组", []],
] as const) {
  assertSame(label, input);
}
console.log("✓ 正确性断言全部通过（myzod 与 zod 结果一致）\n");

// ---------------------------------------------------------------------------
// Benchmark
// ---------------------------------------------------------------------------

group("Nested · valid", () => {
  summary(() => {
    bench("zod · valid", () => do_not_optimize(ZodNested.safeParse(valid)))
      .baseline(true)
      .gc("once");

    bench("myzod · valid", () =>
      do_not_optimize(MyNested.safeParse(valid)),
    ).gc("once");
  });
});

group("Nested · sparse + 未知字段", () => {
  summary(() => {
    bench("zod · sparse", () =>
      do_not_optimize(ZodNested.safeParse(sparseWithExtras)),
    )
      .baseline(true)
      .gc("once");

    bench("myzod · sparse", () =>
      do_not_optimize(MyNested.safeParse(sparseWithExtras)),
    ).gc("once");
  });
});

// 注意：zod 会收集全部 issues，myzod 是 fail-fast 只报第一个错，这一组不是等价语义
group("Nested · invalid（zod 收集全部错误 / myzod fail-fast，非等价）", () => {
  summary(() => {
    bench("zod · invalid", () => do_not_optimize(ZodNested.safeParse(invalid)))
      .baseline(true)
      .gc("once");

    bench("myzod · invalid", () =>
      do_not_optimize(MyNested.safeParse(invalid)),
    ).gc("once");
  });
});

// 诚实性用例：myzod 的 codegen 是一次性成本，这里把它单独暴露出来
group("schema 构建 + 首次 safeParse（一次性成本）", () => {
  summary(() => {
    bench("zod 构建", () => {
      const S = z.object({
        id: z.string(),
        profile: z.object({
          user: z.object({
            name: z.string().min(1),
            email: z.email(),
            age: z.number().int().min(18),
          }),
          address: z.object({
            city: z.string(),
            zip: z.string().length(6),
            geo: z.object({ lat: z.number(), lng: z.number() }).optional(),
          }),
        }),
        tags: z.array(z.string()).default([]),
        role: z.enum(["admin", "user", "guest"]).default("user"),
      });
      do_not_optimize(S.safeParse(valid));
    })
      .baseline(true)
      .gc("once");

    bench("myzod 构建 + 编译", () => {
      do_not_optimize(makeMyNested().safeParse(valid));
    }).gc("once");
  });
});

// ---------------------------------------------------------------------------
// 运行
// ---------------------------------------------------------------------------

const argv = process.argv.slice(2);
const asJson = argv.includes("--json");
const filterArg = argv.find((a) => a.startsWith("--filter="))?.slice(9);

const printCounters = startCounters();

await run({
  format: asJson ? "json" : "mitata",
  ...(filterArg ? { filter: new RegExp(filterArg) } : {}),
});

if (!asJson) await printCounters();
