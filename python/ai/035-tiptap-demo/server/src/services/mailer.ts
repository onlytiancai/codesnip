/**
 * 发邮件（Resend）。
 *
 * **不用 resend SDK** —— server 侧一直是零第三方依赖（node:sqlite、模板渲染器都是自己写的），
 * Resend 的接口就一个 POST，直接 fetch 就行。
 * （官方 SDK 的写法是 `new Resend(key).emails.send({from, to, subject, html})`，
 *  请求体和下面完全一样。）
 *
 * 配置在 `server/.env`：
 *   RESEND_API_KEY=re_xxx
 *   RESEND_FROM=翻译工作台 <noreply@ihuhao.com>
 *
 * ⚠️ 发信域名必须在 Resend 后台**验证过**并配好 DNS（SPF / DKIM），
 * 否则 Resend 会返回 "domain is not verified"。
 */

import { debugLog } from '../utils/debug.js'

const RESEND_ENDPOINT = 'https://api.resend.com/emails'

/** 默认发件人（可在 .env 的 RESEND_FROM 里改）。 */
const DEFAULT_FROM = '翻译工作台 <noreply@ihuhao.com>'

export interface MailContent {
  /** 收件人 */
  to: string
  subject: string
  /** HTML 正文 */
  html: string
  /** 纯文本正文（兜底，客户端不渲染 HTML 时用） */
  text: string
}

/** 服务端配了发信 API Key 吗。 */
export function mailerConfigured(): boolean {
  return Boolean(process.env.RESEND_API_KEY?.trim())
}

/** 发件人地址。 */
export function mailerFrom(): string {
  return process.env.RESEND_FROM?.trim() || DEFAULT_FROM
}

/**
 * 发一封信。失败抛 Error（调用方负责转成合适的 HTTP 状态码）。
 */
export async function sendMail(content: MailContent): Promise<{ id: string }> {
  const apiKey = process.env.RESEND_API_KEY?.trim()
  if (!apiKey) {
    throw new Error('服务端没有配置 RESEND_API_KEY，发不了邮件')
  }
  const from = mailerFrom()

  const res = await fetch(RESEND_ENDPOINT, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      from,
      to: [content.to],
      subject: content.subject,
      html: content.html,
      text: content.text,
    }),
  })

  const body = (await res.json().catch(() => ({}))) as {
    id?: string
    message?: string
    name?: string
  }

  if (!res.ok) {
    const detail = body.message || body.name || `HTTP ${res.status}`
    debugLog('[mail]', '发送失败:', content.to, '-', detail)
    throw new Error(`邮件发送失败：${detail}`)
  }

  debugLog('[mail]', '已发送:', content.to, '|', content.subject, '| id:', body.id)
  return { id: String(body.id ?? '') }
}

/** 验证码邮件的正文。 */
export function verificationMail(opts: {
  code: string
  minutes: number
  purpose: 'register' | 'reset'
}): MailContent {
  const what =
    opts.purpose === 'register' ? '注册翻译工作台账号' : '重置翻译工作台的密码'
  const subject = `【翻译工作台】验证码：${opts.code}`
  const text = `你正在${what}，验证码是 ${opts.code}，${opts.minutes} 分钟内有效。\n如果这不是你本人的操作，忽略这封信就行。`
  const html = `
<div style="font-family:-apple-system,'PingFang SC','Microsoft YaHei',sans-serif;line-height:1.7;color:#222">
  <h2 style="margin:0 0 12px;font-size:18px">翻译工作台</h2>
  <p>你正在${what}，请在 <b>${opts.minutes} 分钟内</b>输入验证码：</p>
  <p style="font-size:30px;font-weight:700;letter-spacing:6px;margin:16px 0;padding:12px 18px;background:#f5f6f8;border-radius:8px;display:inline-block">${opts.code}</p>
  <p style="color:#666;font-size:13px">如果这不是你本人的操作，忽略这封信就行。</p>
</div>`.trim()
  return { to: '', subject, html, text }
}
