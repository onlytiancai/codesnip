---
theme: default
title: TypeScript Generics
lineNumbers: false
---

<!-- narrate 1: 大家好，今天我们一起学习 TypeScript 泛型入门。 -->
<!-- narrate 2: 通过本次分享，你会掌握泛型的核心概念和常见用法。 -->
<!-- narrate 3: 主要内容包括泛型函数、接口、类、类型约束以及多个类型参数。 -->

# TypeScript 泛型入门

<div v-click="1" class="rise" :data-anim-ms="500">

## 参数化类型，让代码更灵活

</div>

<div v-click="2" class="rise" :data-anim-ms="400">

`#泛型`  `#TypeScript`  `#前端`

</div>

<style global>
.slidev-layout {
  text-align: center;
}
@keyframes titleEnter {
  from {
    opacity: 0;
    transform: translateY(24px) scale(0.94);
    filter: blur(8px);
  }
  to {
    opacity: 1;
    transform: translateY(0) scale(1);
    filter: blur(0);
  }
}
.slidev-layout h1 {
  font-size: 3.5rem;
  background: linear-gradient(135deg, #2563eb 0%, #d946ef 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  animation: titleEnter 900ms cubic-bezier(0.22, 1, 0.36, 1) both;
}
.slidev-layout h2 {
  font-size: 1.6rem;
  color: #64748b;
  margin-top: 1.2rem;
}
.slidev-layout .tags {
  margin-top: 2rem;
  font-size: 1.1rem;
  color: #9333ea;
  letter-spacing: 0.15em;
}
.rise.slidev-vclick-target {
  transition: opacity 500ms cubic-bezier(0.22, 1, 0.36, 1),
              transform 500ms cubic-bezier(0.22, 1, 0.36, 1);
}
.rise.slidev-vclick-hidden {
  opacity: 0;
  transform: translateY(24px);
  pointer-events: none;
}
</style>

---

# 什么是泛型

<!-- narrate 1: 我们先从最核心的问题开始：什么是泛型？泛型允许你创建参数化类型，在定义函数、接口、类时不指定具体类型，而在使用时再确定。 -->

<div class="hl-text" :class="{ running: $clicks >= 1 }" v-click="1">

泛型允许你创建**参数化类型**，在定义函数、接口、类时不指定具体类型，而在使用时再确定

</div>

<style global>
.sig {
  background: #f6f8fa;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 14px 18px;
  font-size: 1.1rem;
  line-height: 1.6;
  font-family: 'Fira Code', ui-monospace, SFMono-Regular, Menlo, monospace;
  color: #1f2937;
  overflow-x: auto;
}
.sig .kw { color: #d946ef; font-weight: 600; }
.sig .fn { color: #2563eb; }
.sig .var { color: #ea580c; }
.sig .tok {
  border-radius: 4px;
  padding: 1px 2px;
  transition: all 200ms ease;
}
.sig .tok.hot {
  background: #fde047;
  box-shadow: 0 0 0 3px #fde04766;
  font-weight: 700;
}
.sig .tok.warm {
  background: #fef9c3;
}
.steps {
  margin-top: 16px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.steps .step {
  font-size: 0.95rem;
  color: #334155;
}
.steps .step code {
  background: #eef2ff;
  color: #4338ca;
  padding: 1px 6px;
  border-radius: 4px;
}
/* 荧光笔高亮：默认可见，点击后从左至右涂色 */
.slidev-layout .hl-text.slidev-vclick-hidden {
  opacity: 1;
  pointer-events: auto;
}
.hl-text {
  display: inline-block;
  padding: 0 6px;
  background-image: linear-gradient(
    transparent 55%,
    rgba(253, 224, 71, 0.55) 55%,
    rgba(253, 224, 71, 0.55) 92%,
    transparent 92%
  );
  background-size: 0% 100%;
  background-repeat: no-repeat;
}
.hl-text.running {
  animation: drawHighlight 1.6s cubic-bezier(0.22, 1, 0.36, 1) forwards;
}
@keyframes drawHighlight {
  from { background-size: 0% 100%; }
  to   { background-size: 100% 100%; }
}
</style>

---

# 基础泛型函数

<!-- narrate 1: 下面用一个最基础的泛型函数 identity 来理解它。别急，我们把这行函数签名拆成三部分，一步一步看。 -->
<!-- narrate 2: 先看尖括号里的大写 T。它是类型参数，相当于一个类型占位符。这里并不指定具体类型，而是等到调用函数时再决定 T 到底是什么。 -->
<!-- narrate 3: 再看参数 value 后面的冒号 T。它声明了参数的类型就是刚才的 T。也就是说，你传入什么类型的值，T 就被推断成什么类型。 -->
<!-- narrate 4: 最后看括号后面的冒号 T，这是返回值类型，同样是那个 T。所以这个函数会原样返回你传入的值，输入和输出的类型始终保持一致。 -->
<!-- narrate 5: 调用时你可以显式指定 T 为 string，也可以让 TypeScript 自动推断。这就是泛型最常见的使用方式。 -->

<div v-click="1">

<pre class="sig"><code><span class="kw">function</span> <span class="fn">identity</span><span class="tok" :class="{ hot: $clicks === 2, warm: $clicks > 2 }">&lt;T&gt;</span>(<span class="var">value</span><span class="tok" :class="{ hot: $clicks === 3, warm: $clicks > 3 }">: T</span>)<span class="tok" :class="{ hot: $clicks === 4, warm: $clicks > 4 }">: T</span> {
  <span class="kw">return</span> <span class="var">value</span>;
}</code></pre>

</div>

<div class="steps">
  <div v-click="2" :data-anim-ms="400" class="step"><code>&lt;T&gt;</code> — <b>类型参数</b>：声明一个类型占位符，调用时才决定具体类型</div>
  <div v-click="3" :data-anim-ms="400" class="step"><code>value: T</code> — <b>参数类型</b>：参数的类型就是 T，传入什么类型 T 就是什么</div>
  <div v-click="4" :data-anim-ms="400" class="step"><code>): T</code> — <b>返回类型</b>：同一个 T，函数原样返回输入，类型保持一致</div>
  <div v-click="5" :data-anim-ms="300" class="step"><code>identity&lt;string&gt;('hi')</code> 或 <code>identity(42)</code> — 显式指定或自动推断 T</div>
</div>

---

# 泛型接口

<!-- narrate 1: 接口也可以参数化。下面看一个最简单的 Box 泛型接口。 -->
<!-- narrate 2: 尖括号里的 T 还是类型参数。注意 value 属性的类型是 T，getValue 的返回值也是 T。 -->
<!-- narrate 3: 右侧是用法。声明 Box 等于 string，传入和返回都是 string 类型，类型完全一致。 -->

<div v-click="1" :data-anim-ms="500">

### 泛型接口

```typescript
interface Box<T> {
  value: T;
  getValue(): T;
}
```

</div>

<div v-click="2" :data-anim-ms="500">

### 类型参数 T 的三个位置

- 属性类型 <code>value: T</code>
- 方法返回值 <code>getValue(): T</code>

</div>

<div v-click="3" :data-anim-ms="500">

### 使用

```typescript
const box: Box<string> = {
  value: 'hello',
  getValue() { return this.value; }
};
```

</div>

---

# 泛型类

<!-- narrate 1: 类同样支持泛型。下面定义一个 Container 类，持有一个 T 类型的 item，并提供 getItem 方法返回它。 -->
<!-- narrate 2: 实例化时指定 T 等于 number，container 只能装数字。TypeScript 会自动检查所有赋值。 -->
<!-- narrate 3: getItem 的返回类型自动推导为 number，调用方拿到的也是 number，完全类型安全。 -->

<div v-click="1" :data-anim-ms="500">

### 泛型类

```typescript
class Container<T> {
  private item: T;
  constructor(item: T) { this.item = item; }
  getItem(): T { return this.item; }
}
```

</div>

<div v-click="2" :data-anim-ms="500">

### 实例化

```typescript
const c = new Container<number>(42);
```

</div>

<div v-click="3" :data-anim-ms="500">

### 调用

```typescript
const n: number = c.getItem();  // 类型自动推导
```

</div>

---

# 类型约束

<!-- narrate 1: 默认情况下，T 可以是任何类型。但有时我们希望限制 T 必须具有某些能力，这时用 extends 关键字。 -->
<!-- narrate 2: 下面的例子要求 T 必须有 length 属性。比如 string、array 都有 length，可以传入。 -->
<!-- narrate 3: 但如果你传入 number，就没有 length 属性，TypeScript 会直接报错，编译不通过。 -->
<!-- narrate 4: 这种约束让我们在函数体内放心地访问 T.length 等属性，类型系统保证不会出错。 -->

<div v-click="1" :data-anim-ms="500">

### 用 `extends` 约束类型

```typescript
interface HasLength { length: number; }

function logLen<T extends HasLength>(x: T): void {
  console.log(x.length);
}
```

</div>

<div v-click="2" :data-anim-ms="500">

### ✅ OK — string 有 length

```typescript
logLen('hello');     // OK
logLen([1, 2, 3]);   // OK
```

</div>

<div v-click="3" :data-anim-ms="500">

### ❌ Error — number 没有 length

```typescript
logLen(42);
//       ~~ 编译报错
```

</div>

<div v-click="4" :data-anim-ms="400">

> 类型约束让我们放心使用 T 的属性，TypeScript 在编译期就帮我们排雷。

</div>

---

# 多个类型参数

<!-- narrate 1: 泛型不限于一个类型参数。pair 函数接收 K 类型的 key 和 V 类型的 value，返回一个元组。 -->
<!-- narrate 2: 调用时可以分别为 K 和 V 传入不同类型，比如 string 和 number，组合出键值对。 -->

<div v-click="1" :data-anim-ms="500">

### 多个类型参数

```typescript
function pair<K, V>(key: K, value: V): [K, V] {
  return [key, value];
}
```

</div>

<div v-click="2" :data-anim-ms="500">

### 调用

```typescript
const p1 = pair('age', 18);       // [string, number]
const p2: [string, number] = pair('h', 1);
```

</div>

---

# 总结

<!-- narrate 1: 最后做个总结。泛型的本质是参数化类型，让函数、接口、类在定义时不绑定具体类型，使用时再指定。常用语法包括函数泛型 T、interface T、class T，以及 extends 约束和多个类型参数 K V。 -->

<div v-click="1">

### 三句话回顾

1. **泛型 = 参数化类型**，定义时不绑定具体类型，使用时再确定
2. 语法覆盖 **函数、接口、类**，以及 **extends 约束** 和 **多类型参数**
3. 泛型让代码 **更通用、更类型安全**，避免 `any` 带来的类型丢失

</div>

---

# 谢谢观看

<!-- narrate 1: 感谢大家观看本次分享，希望对你理解 TypeScript 泛型有所帮助。 -->
<!-- narrate 2: 接下来是问答环节，欢迎大家提出问题。 -->

<div v-click="1" :data-anim-ms="800" class="thanks">

## 谢谢观看 🎉

</div>

<div v-click="2" :data-anim-ms="600" class="thanks">

### Q & A

</div>

<style global>
.slidev-layout {
  text-align: center;
}
.thanks h2 {
  font-size: 4rem;
  background: linear-gradient(135deg, #2563eb 0%, #d946ef 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  padding-top: 18vh;
}
.thanks h3 {
  font-size: 2.2rem;
  color: #64748b;
  margin-top: 1.5rem;
  letter-spacing: 0.3em;
}
</style>