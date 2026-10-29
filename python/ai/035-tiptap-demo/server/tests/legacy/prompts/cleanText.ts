/**
 * 模型「输出纯文本」时的通用清洗。
 *
 * 小模型很爱把纯文本包成 ``` 围栏，或者加一句 "English: " 前缀，
 * 或者顺手给结果套上引号。这里统一收拾掉。
 *
 * 翻译、整句改写、指令改写三处共用。
 */
export function cleanText(raw: string): string {
  let text = raw.trim()
  if (!text) throw new Error('empty response')

  // 1. 剥 ``` ... ``` 围栏（带不带语言标记都吃）
  const fenced = text.match(/```(?:[a-z]*\n)?([\s\S]*?)```/i)
  if (fenced) {
    text = fenced[1].trim()
  }

  // 2. 剥常见前缀
  text = text.replace(
    /^(translation|english|rewritten|rewrite|result|answer|output|英文|翻译|改写|结果|输出)\s*[:：]\s*/i,
    '',
  )

  // 3. 剥首尾成对引号
  if (
    (text.startsWith('"') && text.endsWith('"')) ||
    (text.startsWith("'") && text.endsWith("'"))
  ) {
    text = text.slice(1, -1).trim()
  }

  if (!text) throw new Error('empty text after parsing')

  return text
}
