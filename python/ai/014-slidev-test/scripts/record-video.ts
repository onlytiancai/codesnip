// scripts/record-video.ts
// 把 Slidev 演示录成 mp4：自动驱动 v-click，每次点击后等当前旁白音频
// 在浏览器内自然播完（hook `Audio.ended`）再点下一次，最后用 ffmpeg
// 把 Playwright 录的 webm 视频和 MediaRecorder 录的 webm 音频合成为 mp4。
//
// 视频保留开头白屏，音频前垫 firstPlayTimeMs 时长静音对齐 MediaRecorder 起点。
// 附带打一条 h1+fonts.ready 的时间戳日志，方便后续手动 ffmpeg 裁剪。
//
// 用法：
//   pnpm record                       # 录到 output/slide-1.mp4（默认 720p）
//   pnpm record --out my-video.mp4
//   pnpm record --width 1920 --height 1080   # 临时切回 1080p
//   pnpm record --keep-server         # 录完保留 dev server 方便调试
//   pnpm record --no-clean            # 复用 build/ 里的中间产物
//   HEADLESS=1 pnpm record            # 用 headless 模式（CI/无 GUI 时，音频可能为空）

import { spawn, type ChildProcess } from 'node:child_process'
import { execFile } from 'node:child_process'
import { promisify } from 'node:util'
import { mkdir, writeFile, rm, rename } from 'node:fs/promises'
import path from 'node:path'
import { chromium, type Browser, type Page } from 'playwright'

const exec = promisify(execFile)

const W = 1280
const H = 720
const FPS = 30
const DEV_PORT = 3030
// Slidev 默认绑 localhost；用 localhost 而不是 127.0.0.1 避免 macOS 上
// IPv6/IPv4 解析顺序导致的 fetch 失败。
const DEV_URL = `http://localhost:${DEV_PORT}`
const BUILD_DIR = 'build'
const OUTPUT_DIR = 'output'

// 调试日志开关
const DEBUG = process.env.DEBUG === '1' || process.argv.includes('--debug')
function dbg(...args: unknown[]) {
  if (DEBUG) console.log('[debug]', ...args)
}

// 注入到每个新文档 / 新页面的脚本，hook Audio 让它同时被扬声器和 MediaRecorder
// 听到。`__playCount` 用来在 Node 端判断「这次按 Space 有没有真的播音频」；
// `__lastAudio.ended` 用来判断音频是否播完。
const INIT_SCRIPT = `
(() => {
  const AC = window.AudioContext || window.webkitAudioContext;
  if (!AC) { window.__audioCtxUnavailable = true; return; }
  try {
    const ctx = new AC();
    const dest = ctx.createMediaStreamDestination();
    const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
      ? 'audio/webm;codecs=opus'
      : (MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : '');
    const recorder = mimeType
      ? new MediaRecorder(dest.stream, { mimeType })
      : new MediaRecorder(dest.stream);
    const chunks = [];
    recorder.ondataavailable = (e) => { if (e.data && e.data.size) chunks.push(e.data); };
    window.__audioCtx = ctx;
    window.__audioRecorder = recorder;
    window.__audioChunks = chunks;
    window.__playCount = 0;
    window.__lastAudio = null;
    // 记录 MediaRecorder 实际启动的时间（用 performance.now，保证只在页面内一致）
    window.__recorderStart = performance.now();
    window.__firstPlayTime = null;
    window.__stopAudioRecording = () => new Promise((resolve) => {
      if (recorder.state === 'inactive') {
        const blob = new Blob(chunks, { type: 'audio/webm' });
        blob.arrayBuffer().then((buf) => resolve(Array.from(new Uint8Array(buf))));
        return;
      }
      recorder.onstop = async () => {
        const blob = new Blob(chunks, { type: 'audio/webm' });
        const buf = new Uint8Array(await blob.arrayBuffer());
        resolve(Array.from(buf));
      };
      try { recorder.stop(); } catch (_) { resolve([]); }
    });

    const OrigAudio = window.Audio;
    function PatchedAudio(src) {
      const a = new OrigAudio(src);
      try {
        const srcNode = ctx.createMediaElementSource(a);
        srcNode.connect(ctx.destination);
        srcNode.connect(dest);
      } catch (_) { /* 已 hook 过则忽略 */ }
      window.__lastAudio = a;
      window.__playCount = (window.__playCount || 0) + 1;
      // 第一次实际播放：记录从 MediaRecorder 启动到现在的偏移（ms）
      if (window.__firstPlayTime === null) {
        window.__firstPlayTime = performance.now() - window.__recorderStart;
      }
      return a;
    }
    PatchedAudio.prototype = OrigAudio.prototype;
    window.Audio = PatchedAudio;

    try { recorder.start(250); } catch (e) { console.warn('MediaRecorder start failed', e); }
  } catch (e) {
    console.warn('Init script failed:', e);
    window.__audioCtxUnavailable = true;
  }
  // 观察「h1 文本 + document.fonts.ready」同时成立的时刻（ms，从 __recorderStart 算起）。
  // 这里放在 INIT_SCRIPT 里作为字符串原样注入页面，避免 tsx/esbuild 注入
  // __name helper 导致 page.evaluate 报 "ReferenceError: __name is not defined"。
  window.__observeSlideReady = () => new Promise((resolve) => {
    const check = () => {
      const h1 = document.querySelector('.slidev-page h1');
      const fontsOk = typeof document.fonts === 'undefined' || document.fonts.status === 'loaded';
      if (h1 && (h1.textContent || '').trim() && fontsOk) {
        resolve(performance.now() - window.__recorderStart);
      } else {
        requestAnimationFrame(check);
      }
    }
    check();
  });
})();
`.trim()

interface Args {
  keepServer: boolean
  noClean: boolean
  out: string
  headless: boolean
  width: number
  height: number
}

function parseArgs(argv: string[]): Args {
  const args: Args = {
    keepServer: false,
    noClean: false,
    out: path.join(OUTPUT_DIR, 'slide-1.mp4'),
    headless: process.env.HEADLESS === '1',
    width: W,
    height: H,
  }
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]
    if (a === '--keep-server') args.keepServer = true
    else if (a === '--no-clean') args.noClean = true
    else if (a === '--out') args.out = argv[++i]
    else if (a === '--width') {
      const v = Number(argv[++i])
      if (!Number.isFinite(v) || v <= 0) throw new Error(`--width 需要正整数，收到 ${argv[i]}`)
      args.width = Math.round(v)
    } else if (a === '--height') {
      const v = Number(argv[++i])
      if (!Number.isFinite(v) || v <= 0) throw new Error(`--height 需要正整数，收到 ${argv[i]}`)
      args.height = Math.round(v)
    } else if (a === '--debug') { /* consumed via DEBUG env / dbg() */ }
  }
  return args
}

async function prepDirs() {
  await rm(BUILD_DIR, { recursive: true, force: true })
  await mkdir(BUILD_DIR, { recursive: true })
  await mkdir(OUTPUT_DIR, { recursive: true })
}

async function startDevServer(): Promise<ChildProcess> {
  const child = spawn('pnpm', ['dev', '--port', String(DEV_PORT)], {
    stdio: ['ignore', 'pipe', 'pipe'],
    env: process.env,
  })
  child.stdout?.on('data', (d) => process.stdout.write(`[slidev] ${d}`))
  child.stderr?.on('data', (d) => process.stderr.write(`[slidev-err] ${d}`))
  return child
}

async function waitForHttp(url: string, timeoutMs = 60_000): Promise<string> {
  // 同一个端口多试几种 host（macOS 上 IPv6/IPv4 解析顺序可能让 127.0.0.1 失败）
  const u = new URL(url)
  const candidates = [u.host, `127.0.0.1:${u.port}`, `[::1]:${u.port}`]
  const start = Date.now()
  let lastErr: unknown
  while (Date.now() - start < timeoutMs) {
    for (const host of candidates) {
      const tryUrl = `${u.protocol}//${host}${u.pathname}`
      try {
        const res = await fetch(tryUrl)
        if (res.ok) {
          console.log(`[waitForHttp] 命中 ${tryUrl} (status ${res.status})`)
          return tryUrl
        }
        lastErr = new Error(`status ${res.status}`)
      } catch (e) {
        lastErr = e
        dbg(`[waitForHttp] ${tryUrl} 失败:`, (e as Error).message)
      }
    }
    await new Promise((r) => setTimeout(r, 500))
  }
  throw new Error(`Timed out waiting for ${url}: ${lastErr}`)
}

async function runFfmpeg(args: string[]): Promise<void> {
  console.log('[ffmpeg]', args.join(' '))
  const { stdout, stderr } = await exec('ffmpeg', args)
  if (stdout) process.stdout.write(`[ffmpeg-out] ${stdout}`)
  if (stderr) process.stderr.write(`[ffmpeg-err] ${stderr}`)
}

async function muxMp4(
  videoIn: string,
  audioIn: string,
  mp4Out: string,
): Promise<void> {
  await runFfmpeg([
    '-y',
    '-i', videoIn,
    '-i', audioIn,
    '-map', '0:v:0', '-map', '1:a:0',
    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '23', '-preset', 'medium',
    '-r', String(FPS),
    '-c:a', 'aac', '-b:a', '192k', '-ar', '48000',
    '-movflags', '+faststart',
    '-shortest',
    mp4Out,
  ])
}

async function probe(file: string): Promise<string> {
  const { stdout } = await exec('ffprobe', [
    '-v', 'error',
    '-show_entries', 'stream=codec_type,codec_name,width,height,r_frame_rate',
    '-show_entries', 'format=duration,size',
    '-of', 'default',
    file,
  ])
  return stdout
}

async function verify(mp4Path: string, audioPath: string): Promise<void> {
  console.log('[verify mp4]')
  console.log(await probe(mp4Path))
  console.log('[verify audio]')
  console.log(await probe(audioPath))
}

async function runClickLoop(page: Page): Promise<{ totalClicks: number }> {
  let prevPlayCount = 0
  let prevClicks = 0
  let totalClicks = 0
  const tLoopStart = Date.now()

  console.log('[loop] start')

  // 主循环：按 Space，遇到「这次没新音频」就 ArrowRight；URL 不再变则退出
  // eslint-disable-next-line no-constant-condition
  while (true) {
    if (Date.now() - tLoopStart > 10 * 60_000) {
      console.warn('[loop] 超时 10 分钟，强制结束')
      break
    }
    if (totalClicks > 200) {
      console.warn('[loop] 已点击 200 次，强制结束（防止死循环）')
      break
    }

    dbg(`[loop] press Space (clicks=${totalClicks}, prevPlayCount=${prevPlayCount}, prevClicks=${prevClicks})`)
    await page.keyboard.press('Space')
    await page.waitForTimeout(250)

    const state = await page.evaluate(() => ({
      playCount: (window).__playCount || 0,
      url: location.href,
      lastAudioSrc: (window).__lastAudio?.currentSrc || null,
      page: (window).__page || 0,
      clicks: (window).__clicks || 0,
      animMs: (window).__lastClickAnimMs || 0,
    }))

    // 关键判断：clicks 是否真的推进了。Slidev 在 click 已达 max 时按 Space
    // 不会增加 clicks，但当前页 [data-anim-ms] 仍然非零 → 不能据此走「等动效」分支。
    // 翻页后 clicks 会重置为 0，所以只看 clicks 增量即可识别「新 click」。
    const clickAdvanced = state.clicks > prevClicks

    if (state.playCount > prevPlayCount) {
      // 本次按键触发了新音频 → 等它播完
      prevPlayCount = state.playCount
      prevClicks = state.clicks
      totalClicks++
      console.log(`[loop] click #${totalClicks} (page=${state.page}, clicks=${state.clicks}, animMs=${state.animMs}ms) 触发音频 (src=${state.lastAudioSrc}), 等待播放完成...`)

      try {
        await page.waitForFunction(
          () => {
            const a = (window).__lastAudio
            return !!a && a.ended === true
          },
          null,
          { timeout: 60_000 },
        )
        console.log(`[loop] click #${totalClicks} 音频结束`)
      } catch (e) {
        console.warn(`[loop] click #${totalClicks} 音频未在 60s 内播完，继续`)
      }
      // 等 v-click 动画落定（取 animMs 与默认 150ms 的较大者）
      const settleMs = Math.max(state.animMs, 150)
      await page.waitForTimeout(settleMs)
    } else if (clickAdvanced && state.animMs > 0) {
      // 无音频但有点击（带 data-anim-ms 的 v-click）→ 等动效落定再点下一次
      prevClicks = state.clicks
      totalClicks++
      console.log(`[loop] click #${totalClicks} (page=${state.page}, clicks=${state.clicks}) 无音频但有动效 ${state.animMs}ms，等动效...`)
      await page.waitForTimeout(state.animMs)
    } else if (clickAdvanced) {
      // 新 click 触发但既无音频也无动效 → 立即推进
      prevClicks = state.clicks
      totalClicks++
      console.log(`[loop] click #${totalClicks} (page=${state.page}, clicks=${state.clicks}) 无音频无动效`)
    } else {
      console.log(`[loop] press Space 未触发新 click (clicks=${state.clicks}, animMs=${state.animMs}ms)，按 ArrowRight`)
      // 本次按键没推进 click → 当前 slide 的 v-click 已点完（或点不动）
      // 按 ArrowRight 切到下一页
      const before = page.url()
      await page.keyboard.press('ArrowRight')
      await page.waitForTimeout(900)
      const after = page.url()
      if (after === before) {
        console.log('[loop] ArrowRight 未改变 URL，已到末页')
        break
      }
      console.log(`[loop] 切到下一页: ${after}`)
      // 切页后 Slidev 把 clicks 重置为 0；不更新 prevClicks → 下次新 click 1 仍 > 0
      await page.waitForTimeout(300)
    }
  }

  console.log(`[loop] done, totalClicks=${totalClicks}`)
  return { totalClicks }
}

async function main() {
  const args = parseArgs(process.argv.slice(2))

  if (!args.noClean) await prepDirs()
  else {
    await mkdir(BUILD_DIR, { recursive: true })
    await mkdir(OUTPUT_DIR, { recursive: true })
  }

  const server = await startDevServer()
  let browser: Browser | null = null
  let exitCode = 0
  try {
    const workingUrl = await waitForHttp(`${DEV_URL}/`)
    console.log('[main] dev server ready at', workingUrl)

    browser = await chromium.launch({
      headless: args.headless,
      args: [
        '--autoplay-policy=no-user-gesture-required',
        '--no-sandbox',
        '--disable-dev-shm-usage',
      ],
    })

    const ctx = await browser.newContext({
      viewport: { width: args.width, height: args.height },
      recordVideo: { dir: BUILD_DIR, size: { width: args.width, height: args.height } },
    })
    console.log(`[main] 分辨率 ${args.width}x${args.height} @ ${FPS}fps`)
    await ctx.addInitScript({ content: INIT_SCRIPT })
    const page = await ctx.newPage()
    page.on('console', (m) => console.log(`[page] ${m.type()}: ${m.text()}`))
    page.on('pageerror', (e) => console.error('[page-error]', e))

    // Slidev 默认 hash 路由：`http://localhost:3030/#/1`
    const targetUrl = `${workingUrl.replace(/\/$/, '')}/#/1`
    console.log('[main] navigating to', targetUrl)
    await page.goto(targetUrl, { waitUntil: 'domcontentloaded' })

    await page.waitForSelector('.slidev-page', { timeout: 30_000 })
    // 等到「第一页标题真的有文本」再进入 click loop，避免在白屏阶段就开始点击
    await page.waitForFunction(
      () => {
        const h1 = document.querySelector('.slidev-page h1')
        return !!h1 && (h1.textContent || '').trim().length > 0
      },
      null,
      { timeout: 30_000 },
    )
    console.log('[main] 第一页标题已渲染')

    // 附带（非阻塞）：观察"h1 文本 + document.fonts.ready"同时成立的时刻，
    // 打一条日志方便后续手动 ffmpeg 裁剪视频。函数定义放在 INIT_SCRIPT 里
    // 避免 tsx/esbuild 注入 __name helper。
    void page.evaluate(() => (window as any).__observeSlideReady()).then((t: number) => {
      console.log(`[main] [参考] h1+fonts.ready 时刻 = ${t.toFixed(0)}ms（从 __recorderStart 算起，可用于 ffmpeg -ss 手动裁剪）`)
    })

    // 主标题出来后再停 1 秒，让画面在点击前稳一会儿；之后才按 Space 触发第一个
    // v-click（出副标题）和第一句旁白。
    console.log('[main] 停顿 1s，等主标题稳住...')
    await page.waitForTimeout(1000)

    const initState = await page.evaluate(() => ({
      hasCtx: !!window.__audioCtx,
      hasRecorder: !!window.__audioRecorder,
      ctxState: window.__audioCtx?.state,
      ctxUnavailable: !!window.__audioCtxUnavailable,
      url: location.href,
    }))
    console.log('[main] init state:', JSON.stringify(initState))

    const { totalClicks } = await runClickLoop(page)
    console.log(`[main] 录制结束，totalClicks=${totalClicks}`)

    // 给最后一帧动画留点时间
    await page.waitForTimeout(500)

    // 收音频，并读取首次播放的偏移（MediaRecorder 启动 → 第一次 audio.play）
    const { audioBytes, firstPlayTimeMs } = await page.evaluate(() => new Promise<{
      audioBytes: number[]
      firstPlayTimeMs: number
    }>((resolve) => {
      const recorder = (window).__audioRecorder
      const chunks = (window).__audioChunks || []
      if (!recorder || recorder.state === 'inactive') {
        const blob = new Blob(chunks, { type: 'audio/webm' })
        blob.arrayBuffer().then((buf) => resolve({
          audioBytes: Array.from(new Uint8Array(buf)),
          firstPlayTimeMs: (window).__firstPlayTime ?? 0,
        }))
        return
      }
      recorder.onstop = async () => {
        const blob = new Blob(chunks, { type: 'audio/webm' })
        const buf = new Uint8Array(await blob.arrayBuffer())
        resolve({
          audioBytes: Array.from(buf),
          firstPlayTimeMs: (window).__firstPlayTime ?? 0,
        })
      }
      try { recorder.stop() } catch (_) {
        resolve({ audioBytes: [], firstPlayTimeMs: 0 })
      }
    }))

    if (!audioBytes || audioBytes.length === 0) {
      throw new Error('MediaRecorder 没拿到任何音频块（音频可能没录到）')
    }
    const audioPath = path.join(BUILD_DIR, 'narration.webm')
    await writeFile(audioPath, Buffer.from(audioBytes))
    console.log(`[main] 音频写入 ${audioPath} (${audioBytes.length} bytes, firstPlayTimeMs=${firstPlayTimeMs}ms)`)

    // 关 page / ctx，拿到 webm 视频路径
    const videoTmp = await page.video()?.path()
    if (!videoTmp) throw new Error('Playwright 没生成 webm 视频')
    await ctx.close()
    const videoPath = path.join(BUILD_DIR, 'raw.webm')
    await rename(videoTmp, videoPath)
    console.log(`[main] 视频写入 ${videoPath}`)

    // 给音频前垫 firstPlayTimeMs 的静音，让 audio 时间轴对齐 MediaRecorder 启动点
    const padSec = Math.max(0, firstPlayTimeMs / 1000)
    const paddedPath = path.join(BUILD_DIR, 'narration-padded.m4a')
    if (padSec < 0.2) {
      // 偏移太小，直接转封装
      await runFfmpeg(['-y', '-i', audioPath, '-c:a', 'aac', '-b:a', '192k', paddedPath])
    } else {
      await runFfmpeg([
        '-y',
        '-f', 'lavfi', '-t', padSec.toFixed(3),
        '-i', 'anullsrc=channel_layout=stereo:sample_rate=48000',
        '-i', audioPath,
        '-filter_complex',
        '[0:a]aresample=48000,aformat=sample_fmts=s16:channel_layouts=stereo[s];' +
          '[1:a]aresample=48000,aformat=sample_fmts=s16:channel_layouts=stereo[n];' +
          '[s][n]concat=n=2:v=0:a=1[a]',
        '-map', '[a]',
        '-c:a', 'aac', '-b:a', '192k',
        paddedPath,
      ])
    }

    // 合成 mp4
    await muxMp4(videoPath, paddedPath, args.out)
    console.log(`[main] mp4 写入 ${args.out}`)

    await verify(args.out, audioPath)
    console.log('[main] 完成 ✓')
  } catch (e) {
    console.error('[main] 出错:', e)
    exitCode = 1
  } finally {
    if (browser) await browser.close().catch(() => {})
    if (!args.keepServer && !server.killed) {
      server.kill('SIGTERM')
      await new Promise((r) => setTimeout(r, 2000))
      if (!server.killed) server.kill('SIGKILL')
    }
  }
  process.exit(exitCode)
}

main()
