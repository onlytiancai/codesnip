---
theme: default
title: TypeScript Generics
lineNumbers: false
---

# TypeScript Generics

<div class="text-center mt-8">
  <span class="text-gray-400">按 <kbd>→</kbd> 或点击逐步显示</span>
</div>

<div v-click>

## 什么是泛型？

泛型允许你创建**参数化类型**，在定义函数、接口、类时不指定具体类型，而在使用时再确定

</div>

<div v-click>

### 基础泛型函数

```ts twoslash
function identity<T>(value: T): T {
  return value;
}
```

</div>
