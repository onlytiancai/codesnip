浏览器的js如何实现简单的文本压缩

在浏览器的 JavaScript 中，实现简单的文本压缩有几种方法。如果你的目标是把文本压缩成更短的字符串，方便存储或传输，可以根据需求选择。

- `CompressionStream`：浏览器原生 gzip/deflate 压缩，适合真实压缩。
- LZ 类算法：例如 LZ-String，适合把文本压缩成可存储在字符串中的形式。
- 自定义字典或重复串编码：代码简单，但只适合特定类型的文本。
- Base64 或 Unicode 编码：只是编码，不是压缩本身。

## 1. 最简单的原生方案：CompressionStream

现代浏览器可以使用 `CompressionStream` 和 `DecompressionStream`，不需要安装库。

下面给你一套可以直接复制到浏览器控制台运行的代码，实现文本 → 压缩 → Base64 字符串，以及完整的反向解压。这里使用 gzip，压缩和解压都是无损的。参考：MDN Compression Streams API。

[image](https://www.google.com/s2/favicons?domain=https://developer.mozilla.org\&sz=32)

MDN+1



### 1. 完整实现

```
// 文本 -> gzip 压缩 -> Base64
async function compressText(text) {
  const input = new TextEncoder().encode(text);

  const stream = new Blob([input])
    .stream()
    .pipeThrough(new CompressionStream("gzip"));

  const buffer = await new Response(stream).arrayBuffer();
  const bytes = new Uint8Array(buffer);

  // 分块转 Base64，避免大数据展开参数导致溢出
  let binary = "";
  const chunkSize = 0x8000;

  for (let i = 0; i < bytes.length; i += chunkSize) {
    binary += String.fromCharCode(
      ...bytes.subarray(i, i + chunkSize)
    );
  }

  return btoa(binary);
}

// Base64 -> gzip 解压 -> 原始文本
async function decompressText(base64) {
  const binary = atob(base64);
  const bytes = Uint8Array.from(
    binary,
    c => c.charCodeAt(0)
  );

  const stream = new Blob([bytes])
    .stream()
    .pipeThrough(new DecompressionStream("gzip"));

  return await new Response(stream).text();
}
```

使用示例：

```
const text = "你好，世界！".repeat(1000);

const compressed = await compressText(text);
const restored = await decompressText(compressed);

console.log("原文字符数：", text.length);
console.log("压缩后 Base64 长度：", compressed.length);
console.log("还原成功：", restored === text);
```

### 2. 如果你追求的是最短的字符串

上面的代码存在一个额外开销：gzip 压缩完以后又转成 Base64，长度会增加约三分之一。

可以根据用途选择：

| 方案             | 优点                        | 缺点            |
| -------------- | ------------------------- | ------------- |
| gzip + Base64  | 兼容性好，方便存储和传输              | Base64 会增加长度  |
| gzip 二进制数据     | 不需要 Base64 的额外开销          | 不能直接作为普通文本使用  |
| LZ-String      | 适合将压缩结果保存为 JavaScript 字符串 | 压缩率依赖文本内容     |
| 自定义 Unicode 编码 | 有机会减少最终字符数量               | 需要自行处理字符兼容和转义 |

特别提醒： 短文本压缩后可能反而更长，因为 gzip 本身有文件头等固定开销。如果你的最终目的是把数据塞进 URL、`localStorage` 或一个尽可能短的 Unicode 字符串，我会采用不同的实现方式。