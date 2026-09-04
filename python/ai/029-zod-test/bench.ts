import { run, bench, group, summary, do_not_optimize } from "mitata";
import { z } from "zod";
import { startCounters } from "./counters.ts";

// ---------------------------------------------------------------------------
// Schema
// ---------------------------------------------------------------------------

const SimpleUser = z.object({
  name: z.string().min(1),
  email: z.email(),
  age: z.number().int().min(18),
});

const Nested = z.object({
  id: z.string(),
  profile: z.object({
    user: SimpleUser,
    address: z.object({
      city: z.string(),
      zip: z.string().length(6),
      geo: z.object({ lat: z.number(), lng: z.number() }).optional(),
    }),
  }),
  tags: z.array(z.string()).default([]),
  role: z.enum(["admin", "user", "guest"]).default("user"),
});

const UserArray = z.array(SimpleUser);

const DiscriminatedUnion = z.discriminatedUnion("kind", [
  z.object({ kind: z.literal("circle"), radius: z.number() }),
  z.object({ kind: z.literal("square"), side: z.number() }),
  z.object({ kind: z.literal("rect"), w: z.number(), h: z.number() }),
]);

const PlainUnion = z.union([
  z.object({ kind: z.literal("circle"), radius: z.number() }),
  z.object({ kind: z.literal("square"), side: z.number() }),
  z.object({ kind: z.literal("rect"), w: z.number(), h: z.number() }),
]);

// ---------------------------------------------------------------------------
// 数据集
// ---------------------------------------------------------------------------

const validUser = { name: "Tom", email: "tom@example.com", age: 20 };
const invalidUser = { name: "", email: "not-an-email", age: 12 };

const validNested = {
  id: "u-1",
  profile: {
    user: validUser,
    address: { city: "Shanghai", zip: "200000", geo: { lat: 31.2, lng: 121.5 } },
  },
  tags: ["a", "b"],
  role: "admin",
};
const invalidNested = {
  id: 1,
  profile: { user: invalidUser, address: { city: "Shanghai", zip: "20" } },
};

const validArray = Array.from({ length: 100 }, (_, i) => ({
  name: `u${i}`,
  email: `u${i}@example.com`,
  age: 18 + (i % 50),
}));
const invalidArray = validArray.map((u, i) =>
  i === 50 ? { ...u, email: "bad" } : u,
);

const validShape = { kind: "rect", w: 3, h: 4 };
const invalidShape = { kind: "rect", w: "3", h: 4 };

// 手写朴素校验，作为 baseline
const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
function manualValidateUser(v: any) {
  if (typeof v !== "object" || v === null) return null;
  if (typeof v.name !== "string" || v.name.length < 1) return null;
  if (typeof v.email !== "string" || !EMAIL_RE.test(v.email)) return null;
  if (typeof v.age !== "number" || !Number.isInteger(v.age) || v.age < 18)
    return null;
  return { name: v.name, email: v.email, age: v.age };
}

// ---------------------------------------------------------------------------
// 用例
// ---------------------------------------------------------------------------

group("simple object", () => {
  summary(() => {
    bench("manual validate (baseline)", () =>
      do_not_optimize(manualValidateUser(validUser)),
    )
      .baseline(true)
      .gc("once");

    bench("parse · valid", () =>
      do_not_optimize(SimpleUser.parse(validUser)),
    ).gc("once");

    bench("safeParse · valid", () =>
      do_not_optimize(SimpleUser.safeParse(validUser)),
    ).gc("once");

    bench("safeParse · invalid", () =>
      do_not_optimize(SimpleUser.safeParse(invalidUser)),
    ).gc("once");

    bench("parse · invalid + try/catch", () => {
      try {
        do_not_optimize(SimpleUser.parse(invalidUser));
      } catch (e) {
        do_not_optimize(e);
      }
    }).gc("once");
  });
});

group("nested object", () => {
  summary(() => {
    bench("safeParse · valid", () =>
      do_not_optimize(Nested.safeParse(validNested)),
    ).gc("once");

    bench("safeParse · invalid", () =>
      do_not_optimize(Nested.safeParse(invalidNested)),
    ).gc("once");
  });
});

group("array of 100", () => {
  summary(() => {
    bench("manual · valid (baseline)", () =>
      do_not_optimize(validArray.map(manualValidateUser)),
    )
      .baseline(true)
      .gc("once");

    bench("safeParse · valid", () =>
      do_not_optimize(UserArray.safeParse(validArray)),
    ).gc("once");

    bench("safeParse · 1 bad element", () =>
      do_not_optimize(UserArray.safeParse(invalidArray)),
    ).gc("once");
  });
});

group("union", () => {
  summary(() => {
    bench("discriminatedUnion · valid", () =>
      do_not_optimize(DiscriminatedUnion.safeParse(validShape)),
    )
      .baseline(true)
      .gc("once");

    bench("union · valid", () =>
      do_not_optimize(PlainUnion.safeParse(validShape)),
    ).gc("once");

    bench("discriminatedUnion · invalid", () =>
      do_not_optimize(DiscriminatedUnion.safeParse(invalidShape)),
    ).gc("once");

    bench("union · invalid", () =>
      do_not_optimize(PlainUnion.safeParse(invalidShape)),
    ).gc("once");
  });
});

group("schema construction", () => {
  summary(() => {
    bench("reuse cached schema (baseline)", () =>
      do_not_optimize(SimpleUser.safeParse(validUser)),
    )
      .baseline(true)
      .gc("once");

    bench("build schema every time", () => {
      const S = z.object({
        name: z.string().min(1),
        email: z.email(),
        age: z.number().int().min(18),
      });
      do_not_optimize(S.safeParse(validUser));
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
