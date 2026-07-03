---
theme: default
title: TypeScript Generics
lineNumbers: false
clicks: 5
---

# TypeScript Generics

<!-- 旁白（narrate N: 显式绑定第 N 个 click）-->
<!-- narrate 1: 欢迎来到 TypeScript 泛型入门。我们先从最核心的问题开始：什么是泛型？泛型允许你创建参数化类型，在定义函数、接口、类时不指定具体类型，而在使用时再确定。 -->
<!-- narrate 2: 下面用一个最基础的泛型函数 identity 来理解它。别急，我们把这行函数签名拆成三部分，一步一步看。 -->
<!-- narrate 3: 先看尖括号里的大写 T。它是类型参数，相当于一个类型占位符。这里并不指定具体类型，而是等到调用函数时再决定 T 到底是什么。 -->
<!-- narrate 4: 再看参数 value 后面的冒号 T。它声明了参数的类型就是刚才的 T。也就是说，你传入什么类型的值，T 就被推断成什么类型。 -->
<!-- narrate 5: 最后看括号后面的冒号 T，这是返回值类型，同样是那个 T。所以这个函数会原样返回你传入的值，输入和输出的类型始终保持一致。 -->


<div v-click="1">

## 什么是泛型？

泛型允许你创建**参数化类型**，在定义函数、接口、类时不指定具体类型，而在使用时再确定

</div>

<div v-click="2">

### 基础泛型函数

<pre class="sig"><code><span class="kw">function</span> <span class="fn">identity</span><span class="tok" :class="{ hot: $clicks === 3, warm: $clicks > 3 }">&lt;T&gt;</span>(<span class="var">value</span><span class="tok" :class="{ hot: $clicks === 4, warm: $clicks > 4 }">: T</span>)<span class="tok" :class="{ hot: $clicks === 5, warm: $clicks > 5 }">: T</span> {
  <span class="kw">return</span> <span class="var">value</span>;
}</code></pre>

</div>

<div class="steps">
  <div v-click="3" class="step"><code>&lt;T&gt;</code> — <b>类型参数</b>：声明一个类型占位符，调用时才决定具体类型</div>
  <div v-click="4" class="step"><code>value: T</code> — <b>参数类型</b>：参数的类型就是 T，传入什么类型 T 就是什么</div>
  <div v-click="5" class="step"><code>): T</code> — <b>返回类型</b>：同一个 T，函数原样返回输入，类型保持一致</div>
</div>

<style>
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
</style>
