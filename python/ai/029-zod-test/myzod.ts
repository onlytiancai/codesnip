// myzod —— 一个只实现 safeParse 的最小化高性能校验器。
//
// 性能来源是 codegen：schema 树在首次 safeParse 时被编译成一个完全内联的特化函数，
// 没有逐字段的闭包分发、没有递归调用。语义上对齐 zod 的两点：
//   1. object 剔除未声明的字段（strip）
//   2. 成功时只在 schema 声明到的层级重建对象，叶子值直接引用（不深拷贝）
// 与 zod 的差异：fail-fast，只报第一个错误。

export type Ok<T> = { success: true; data: T };
export type Err = { success: false; error: { path: string; message: string } };
export type Result<T> = Ok<T> | Err;

type Node =
  | { kind: "string"; min?: number; length?: number }
  | { kind: "email" }
  | { kind: "number"; int?: boolean; min?: number }
  | { kind: "object"; shape: Record<string, Schema> }
  | { kind: "array"; inner: Schema }
  | { kind: "enum"; values: string[] };

const EMAIL_RE =
  /^(?!\.)(?!.*\.\.)([a-z0-9_'+\-.]*)[a-z0-9_+-]@([a-z0-9][a-z0-9-]*\.)+[a-z]{2,}$/i;

// ---------------------------------------------------------------------------
// Schema 节点：纯描述符 + 原型方法，编译前不产生任何闭包
// ---------------------------------------------------------------------------

let uid = 0;

export class Schema<T = any> {
  node: Node;
  isOptional = false;
  hasDefault = false;
  defaultValue: unknown;
  #compiled: ((input: unknown) => Result<T>) | null = null;

  constructor(node: Node) {
    this.node = node;
  }

  min(n: number): this {
    const k = this.node.kind;
    if (k !== "string" && k !== "number")
      throw new TypeError(`.min() 不适用于 ${k}`);
    (this.node as any).min = n;
    return this;
  }

  length(n: number): this {
    if (this.node.kind !== "string") throw new TypeError(".length() 只适用于 string");
    this.node.length = n;
    return this;
  }

  int(): this {
    if (this.node.kind !== "number") throw new TypeError(".int() 只适用于 number");
    this.node.int = true;
    return this;
  }

  optional(): Schema<T | undefined> {
    this.isOptional = true;
    return this as any;
  }

  // 注意：与 zod 行为一致，非函数的 default 值不会被克隆——每次校验返回的是
  // 同一个实例。对 [] / {} 这类可变默认值，调用方改动它会影响后续所有结果。
  default(v: T): this {
    this.hasDefault = true;
    this.defaultValue = v;
    return this;
  }

  safeParse(input: unknown): Result<T> {
    return (this.#compiled ??= compile(this))(input);
  }

  /** 供调试：查看生成的源码 */
  source(): string {
    return build(this).src;
  }
}

// ---------------------------------------------------------------------------
// 编译器
// ---------------------------------------------------------------------------

interface Ctx {
  consts: unknown[];
  lines: string[];
}

/** 把一个运行时常量放进闭包数组，返回它的引用表达式 */
function konst(ctx: Ctx, v: unknown): string {
  return `C[${ctx.consts.push(v) - 1}]`;
}

function fail(path: string, msg: string): string {
  return `return{success:false,error:{path:${JSON.stringify(
    path,
  )},message:${JSON.stringify(msg)}}}`;
}

/**
 * 为 `src`（一个已求值的输入表达式）生成校验代码，并返回**结果值的表达式**。
 * path 在编译期就拼好，运行时零开销。
 */
function gen(ctx: Ctx, schema: Schema, src: string, path: string): string {
  const node = schema.node;
  const v = `v${uid++}`;
  ctx.lines.push(`const ${v}=${src};`);

  // optional / default 的前置处理
  let guardOpen = false;
  let out = v;
  if (schema.hasDefault) {
    const d = konst(ctx, schema.defaultValue);
    const r = `r${uid++}`;
    ctx.lines.push(`let ${r}=${d};`);
    ctx.lines.push(`if(${v}!==undefined){`);
    guardOpen = true;
    out = r;
  } else if (schema.isOptional) {
    const r = `r${uid++}`;
    ctx.lines.push(`let ${r}=undefined;`);
    ctx.lines.push(`if(${v}!==undefined){`);
    guardOpen = true;
    out = r;
  }

  const assign = (expr: string) =>
    guardOpen ? ctx.lines.push(`${out}=${expr};`) : (out = expr);

  switch (node.kind) {
    case "string": {
      ctx.lines.push(`if(typeof ${v}!=="string"){${fail(path, "expected string")}}`);
      if (node.length !== undefined)
        ctx.lines.push(
          `if(${v}.length!==${node.length}){${fail(
            path,
            `expected string of length ${node.length}`,
          )}}`,
        );
      if (node.min !== undefined)
        ctx.lines.push(
          `if(${v}.length<${node.min}){${fail(
            path,
            `expected at least ${node.min} characters`,
          )}}`,
        );
      assign(v);
      break;
    }

    case "email": {
      ctx.lines.push(`if(typeof ${v}!=="string"){${fail(path, "expected string")}}`);
      ctx.lines.push(
        `if(!${konst(ctx, EMAIL_RE)}.test(${v})){${fail(path, "invalid email")}}`,
      );
      assign(v);
      break;
    }

    case "number": {
      // 排除 NaN：typeof NaN === "number"
      ctx.lines.push(
        `if(typeof ${v}!=="number"||${v}!==${v}){${fail(path, "expected number")}}`,
      );
      if (node.int)
        ctx.lines.push(
          `if(!Number.isInteger(${v})){${fail(path, "expected integer")}}`,
        );
      if (node.min !== undefined)
        ctx.lines.push(
          `if(${v}<${node.min}){${fail(path, `expected >= ${node.min}`)}}`,
        );
      assign(v);
      break;
    }

    case "enum": {
      const test = node.values.map((s) => `${v}!==${JSON.stringify(s)}`).join("&&");
      ctx.lines.push(
        `if(${test}){${fail(path, `expected one of ${node.values.join(", ")}`)}}`,
      );
      assign(v);
      break;
    }

    case "object": {
      ctx.lines.push(
        `if(typeof ${v}!=="object"||${v}===null||Array.isArray(${v})){${fail(
          path,
          "expected object",
        )}}`,
      );
      // 逐字段内联展开，最后用对象字面量一次性构造 —— 单形态，且天然 strip 未知字段
      const required: string[] = [];
      const opt: { key: string; val: string }[] = [];
      for (const [key, sub] of Object.entries(node.shape)) {
        const childPath = path ? `${path}.${key}` : key;
        const val = gen(ctx, sub, `${v}[${JSON.stringify(key)}]`, childPath);
        // 有 default 的字段一定有值，键恒存在；纯 optional 字段缺失时要整个不出现，
        // 而不是留一个 `key: undefined`（zod 会省略该键，deepStrictEqual 区分二者）
        if (sub.isOptional && !sub.hasDefault) opt.push({ key, val });
        else required.push(`${JSON.stringify(key)}:${val}`);
      }

      if (opt.length === 0) {
        assign(`{${required.join(",")}}`);
      } else if (opt.length <= 2) {
        // 展开成 2^n 个完整字面量，每个分支的隐藏类都是固定的，避免建完再挂属性
        const branch = (i: number, acc: string[]): string => {
          if (i === opt.length) return `{${required.concat(acc).join(",")}}`;
          const { key, val } = opt[i];
          const withKey = branch(i + 1, acc.concat(`${JSON.stringify(key)}:${val}`));
          const without = branch(i + 1, acc);
          return `(${val}!==undefined?${withKey}:${without})`;
        };
        assign(branch(0, []));
      } else {
        // optional 字段过多，2^n 会爆炸，退回“先建再挂”
        const o = `o${uid++}`;
        ctx.lines.push(`const ${o}={${required.join(",")}};`);
        for (const { key, val } of opt)
          ctx.lines.push(
            `if(${val}!==undefined)${o}[${JSON.stringify(key)}]=${val};`,
          );
        assign(o);
      }
      break;
    }

    case "array": {
      ctx.lines.push(`if(!Array.isArray(${v})){${fail(path, "expected array")}}`);
      const arr = `a${uid++}`;
      const i = `i${uid++}`;
      ctx.lines.push(`const ${arr}=new Array(${v}.length);`);
      ctx.lines.push(`for(let ${i}=0;${i}<${v}.length;${i}++){`);
      const el = gen(ctx, node.inner, `${v}[${i}]`, `${path}[]`);
      ctx.lines.push(`${arr}[${i}]=${el};`);
      ctx.lines.push(`}`);
      assign(arr);
      break;
    }
  }

  if (guardOpen) ctx.lines.push(`}`);
  return out;
}

function build(schema: Schema): { src: string; consts: unknown[] } {
  uid = 0;
  const ctx: Ctx = { consts: [], lines: [] };
  const result = gen(ctx, schema, "input", "");
  const src = `return function(input){\n${ctx.lines.join("\n")}\nreturn{success:true,data:${result}};\n}`;
  return { src, consts: ctx.consts };
}

function compile<T>(schema: Schema<T>): (input: unknown) => Result<T> {
  const { src, consts } = build(schema);
  return new Function("C", src)(consts);
}

// ---------------------------------------------------------------------------
// 公开构造器
// ---------------------------------------------------------------------------

export const m = {
  string: () => new Schema<string>({ kind: "string" }),
  email: () => new Schema<string>({ kind: "email" }),
  number: () => new Schema<number>({ kind: "number" }),
  array: <T>(inner: Schema<T>) => new Schema<T[]>({ kind: "array", inner }),
  object: <S extends Record<string, Schema>>(shape: S) =>
    new Schema<{ [K in keyof S]: S[K] extends Schema<infer U> ? U : never }>({
      kind: "object",
      shape,
    }),
  enum: <const V extends readonly string[]>(values: V) =>
    new Schema<V[number]>({ kind: "enum", values: values as unknown as string[] }),
};
