/**
 * 程序化行星纹理生成器。
 *
 * 全部纹理用 canvas 2D 生成：值噪声 fBm + 域扭曲做基础表面，
 * 再用 canvas 矢量叠加陨石坑/极冠/大红斑等特征。
 * 零网络依赖、永不加载失败、体积 ~0KB。
 *
 * 等距柱状投影（2:1），噪声在水平方向做周期性无缝拼接。
 */

import * as THREE from 'three/webgpu'

// ---------- 基础工具 ----------

/** mulberry32 种子随机数 */
function rng(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a |= 0
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function smoothstep(t: number): number {
  return t * t * (3 - 2 * t)
}

/**
 * 周期值噪声 fBm（水平方向无缝）。
 * 返回 Float32Array，长度 w*h，值域约 [0, 1]。
 */
function fbm2(
  rand: () => number,
  w: number,
  h: number,
  octaves: number,
  baseFreq: number,
  opts: { warp?: (x: number, y: number) => [number, number]; gain?: number } = {},
): Float32Array {
  const gain = opts.gain ?? 0.5
  const out = new Float32Array(w * h)
  // 每个八度一张周期点阵
  const lattices: Float32Array[] = []
  let amp = 1
  let totalAmp = 0
  for (let o = 0; o < octaves; o++) {
    const nx = Math.max(2, Math.round(baseFreq * 2 ** o))
    const ny = Math.max(2, Math.round((nx * h) / (2 * w)))
    const lattice = new Float32Array(nx * ny)
    for (let i = 0; i < lattice.length; i++) lattice[i] = rand()
    lattices.push(lattice)
    totalAmp += amp
    amp *= gain
  }
  const warp = opts.warp
  for (let y = 0; y < h; y++) {
    const v = y / (h - 1)
    for (let x = 0; x < w; x++) {
      const u = x / w
      let sx = u
      let sy = v
      if (warp) [sx, sy] = warp(u, v)
      let sum = 0
      let a = 1
      for (let o = 0; o < octaves; o++) {
        const lattice = lattices[o]
        const nx = Math.max(2, Math.round(baseFreq * 2 ** o))
        const ny = Math.max(2, Math.round((nx * h) / (2 * w)))
        const px = sx * nx
        const py = sy * ny
        const ix = Math.floor(px)
        const iy = Math.floor(py)
        const fx = smoothstep(px - ix)
        const fy = smoothstep(py - iy)
        // 水平周期采样
        const x0 = ((ix % nx) + nx) % nx
        const x1 = (x0 + 1) % nx
        const y0 = Math.min(Math.max(iy, 0), ny - 1)
        const y1 = Math.min(y0 + 1, ny - 1)
        const n00 = lattice[y0 * nx + x0]
        const n10 = lattice[y0 * nx + x1]
        const n01 = lattice[y1 * nx + x0]
        const n11 = lattice[y1 * nx + x1]
        const top = n00 + (n10 - n00) * fx
        const bot = n01 + (n11 - n01) * fx
        sum += (top + (bot - top) * fy) * a
        a *= gain
      }
      out[y * w + x] = sum / totalAmp
    }
  }
  return out
}

/** 颜色插值工具 */
type Stop = [number, [number, number, number]]
function ramp(stops: Stop[], t: number): [number, number, number] {
  const c = Math.min(1, Math.max(0, t))
  for (let i = 0; i < stops.length - 1; i++) {
    const [t0, c0] = stops[i]
    const [t1, c1] = stops[i + 1]
    if (c <= t1) {
      const k = t1 === t0 ? 0 : (c - t0) / (t1 - t0)
      return [
        c0[0] + (c1[0] - c0[0]) * k,
        c0[1] + (c1[1] - c0[1]) * k,
        c0[2] + (c1[2] - c0[2]) * k,
      ]
    }
  }
  return stops[stops.length - 1][1]
}

function makeCanvas(w: number, h: number): [HTMLCanvasElement, CanvasRenderingContext2D] {
  const canvas = document.createElement('canvas')
  canvas.width = w
  canvas.height = h
  const ctx = canvas.getContext('2d')!
  return [canvas, ctx]
}

/** 用逐像素着色函数生成纹理 */
function paintTexture(
  w: number,
  h: number,
  colorize: (u: number, v: number, x: number, y: number) => [number, number, number, number],
): [HTMLCanvasElement, ImageData] {
  const [canvas, ctx] = makeCanvas(w, h)
  const img = ctx.createImageData(w, h)
  const data = img.data
  for (let y = 0; y < h; y++) {
    const v = y / (h - 1)
    for (let x = 0; x < w; x++) {
      const u = x / w
      const [r, g, b, a] = colorize(u, v, x, y)
      const i = (y * w + x) * 4
      data[i] = r
      data[i + 1] = g
      data[i + 2] = b
      data[i + 3] = a
    }
  }
  ctx.putImageData(img, 0, 0)
  return [canvas, img]
}

function toTexture(canvas: HTMLCanvasElement, srgb = true): THREE.CanvasTexture {
  const tex = new THREE.CanvasTexture(canvas)
  tex.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace
  tex.wrapS = THREE.RepeatWrapping
  tex.wrapT = THREE.ClampToEdgeWrapping
  tex.anisotropy = 8 // 环等斜视角表面依赖各向异性过滤
  return tex
}

/** 在画布上画陨石坑 */
function drawCraters(
  ctx: CanvasRenderingContext2D,
  rand: () => number,
  w: number,
  h: number,
  count: number,
  opts: { maxR?: number; alpha?: number } = {},
): void {
  const maxR = opts.maxR ?? 0.06
  ctx.save()
  ctx.globalAlpha = opts.alpha ?? 0.55
  for (let i = 0; i < count; i++) {
    const cx = rand() * w
    const cy = h * (0.12 + rand() * 0.76) // 避开极区
    const r = Math.max(2, maxR * w * (0.2 + rand() * 0.8))
    const g = ctx.createRadialGradient(cx, cy, r * 0.1, cx, cy, r)
    g.addColorStop(0, 'rgba(20,16,14,0.85)')
    g.addColorStop(0.55, 'rgba(30,25,20,0.5)')
    g.addColorStop(0.78, 'rgba(255,240,220,0.32)') // 亮边
    g.addColorStop(1, 'rgba(0,0,0,0)')
    ctx.fillStyle = g
    ctx.beginPath()
    ctx.arc(cx, cy, r, 0, Math.PI * 2)
    ctx.fill()
    // 小坑内的小坑
    if (rand() > 0.7) {
      const g2 = ctx.createRadialGradient(cx, cy, 0, cx, cy, r * 0.45)
      g2.addColorStop(0, 'rgba(15,12,10,0.8)')
      g2.addColorStop(1, 'rgba(0,0,0,0)')
      ctx.fillStyle = g2
      ctx.beginPath()
      ctx.arc(cx + r * 0.1, cy - r * 0.1, r * 0.45, 0, Math.PI * 2)
      ctx.fill()
    }
  }
  ctx.restore()
}

/** 画椭圆斑（大红斑/海王星暗斑等） */
function drawOvalSpot(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  rx: number,
  ry: number,
  rot: number,
  stops: [number, string][],
  alpha = 0.9,
): void {
  ctx.save()
  ctx.globalAlpha = alpha
  ctx.translate(x, y)
  ctx.rotate(rot)
  const g = ctx.createRadialGradient(0, 0, 0, 0, 0, Math.max(rx, ry))
  for (const [t, color] of stops) g.addColorStop(t, color)
  ctx.fillStyle = g
  ctx.beginPath()
  ctx.ellipse(0, 0, rx, ry, 0, 0, Math.PI * 2)
  ctx.fill()
  ctx.restore()
}

// ---------- 各天体生成器 ----------

export function sunTexture(w: number): THREE.CanvasTexture {
  const h = w / 2
  const rand = rng(11)
  const n = fbm2(rand, w, h, 6, 6, {
    warp: (u, v) => [u + (fbmWarp(rand2, u, v, w, h) - 0.5) * 0.08, v + 0.0],
  })
  const [canvas] = paintTexture(w, h, (_u, _v, x, y) => {
    const t = n[y * w + x]
    const [r, g, b] = ramp(
      [
        [0, [255, 140, 20]],
        [0.35, [255, 190, 60]],
        [0.6, [255, 224, 110]],
        [0.82, [255, 244, 190]],
        [1, [255, 255, 244]],
      ],
      t,
    )
    return [r, g, b, 255]
  })
  return toTexture(canvas)
}

// 小工具：域扭曲用的单层噪声（按随机函数区分缓存）
const rand2 = rng(7)
const warpCache = new WeakMap<() => number, Map<number, Float32Array>>()
function fbmWarp(rand: () => number, u: number, v: number, w: number, h: number): number {
  let bySize = warpCache.get(rand)
  if (!bySize) {
    bySize = new Map()
    warpCache.set(rand, bySize)
  }
  let grid = bySize.get(w)
  if (!grid) {
    grid = fbm2(rand, w, h, 3, 3)
    bySize.set(w, grid)
  }
  const x = Math.min(w - 1, Math.round(u * w))
  const y = Math.min(h - 1, Math.round(v * h))
  return grid[y * w + x]
}

export function mercuryTexture(w: number): THREE.CanvasTexture {
  const h = w / 2
  const rand = rng(21)
  const n = fbm2(rand, w, h, 5, 5)
  const [canvas] = paintTexture(w, h, (_u, _v, x, y) => {
    const t = n[y * w + x]
    const [r, g, b] = ramp(
      [
        [0, [96, 90, 86]],
        [0.45, [140, 134, 128]],
        [0.7, [172, 166, 158]],
        [1, [198, 192, 184]],
      ],
      t,
    )
    return [r, g, b, 255]
  })
  const ctx = canvas.getContext('2d')!
  drawCraters(ctx, rand, w, h, Math.round(w / 8), { maxR: 0.09, alpha: 0.6 })
  drawCraters(ctx, rand, w, h, Math.round(w / 3), { maxR: 0.025, alpha: 0.45 })
  return toTexture(canvas)
}

export function venusTexture(w: number): THREE.CanvasTexture {
  const h = w / 2
  const rand = rng(31)
  // 域扭曲制造漩涡状云带
  const warp = fbm2(rand, w, h, 3, 3)
  const n = fbm2(rand, w, h, 5, 4, {
    warp: (u, v) => {
      const x = Math.min(w - 1, Math.round(u * w))
      const y = Math.min(h - 1, Math.round(v * h))
      const d = warp[y * w + x]
      return [u + (d - 0.5) * 0.25, v + (d - 0.5) * 0.06]
    },
  })
  const [canvas] = paintTexture(w, h, (_u, v, x, y) => {
    const t = n[y * w + x]
    // 云带：暗金 → 亮黄白
    const band = Math.sin(v * Math.PI * 9 + t * 4) * 0.5 + 0.5
    const [r, g, b] = ramp(
      [
        [0, [206, 152, 78]],
        [0.35, [236, 196, 120]],
        [0.65, [248, 224, 160]],
        [1, [255, 240, 200]],
      ],
      t * 0.55 + band * 0.45,
    )
    return [r, g, b, 255]
  })
  return toTexture(canvas)
}

export function earthTexture(w: number): THREE.CanvasTexture {
  const h = w / 2
  const rand = rng(41)
  const continents = fbm2(rand, w, h, 6, 4, { gain: 0.55 })
  const detail = fbm2(rand, w, h, 4, 8)
  const [canvas] = paintTexture(w, h, (_u, v, x, y) => {
    const i = y * w + x
    const t = continents[i]
    const d = detail[i]
    const lat = Math.abs(v - 0.5) * 2 // 0 赤道 → 1 极
    // 极地冰盖
    if (lat > 0.88 + d * 0.08) return [240, 246, 252, 255]
    const seaLevel = 0.54
    if (t < seaLevel) {
      // 海洋：近岸浅蓝 → 深海深蓝
      const depth = (seaLevel - t) / seaLevel
      const [r, g, b] = ramp(
        [
          [0, [38, 96, 160]],
          [0.25, [30, 84, 150]],
          [1, [16, 44, 100]],
        ],
        depth,
      )
      return [r, g, b, 255]
    }
    // 陆地：纬度 + 噪声决定颜色
    const landH = (t - seaLevel) / (1 - seaLevel) // 0 海岸 → 1 内陆
    let [r, g, b]: [number, number, number] = [90, 130, 60]
    if (lat < 0.28) {
      ;[r, g, b] = ramp(
        [
          [0, [96, 148, 66]],
          [0.6, [74, 130, 60]],
          [1, [112, 96, 52]],
        ],
        landH + d * 0.5,
      )
    } else if (lat < 0.45) {
      ;[r, g, b] = ramp(
        [
          [0, [150, 142, 92]],
          [0.5, [188, 162, 104]],
          [1, [172, 138, 90]],
        ],
        landH + d * 0.6,
      )
    } else {
      ;[r, g, b] = ramp(
        [
          [0, [110, 140, 84]],
          [0.55, [140, 130, 96]],
          [1, [200, 196, 176]],
        ],
        landH + d * 0.5,
      )
    }
    return [r, g, b, 255]
  })
  return toTexture(canvas)
}

export function earthCloudTexture(w: number): THREE.CanvasTexture {
  const h = w / 2
  const rand = rng(43)
  const warpRand = rng(44) // 注意：必须在 fbm2 外创建一次，否则每个像素新建 rng → 缓存失效 → 每像素重算整张噪声图
  const n = fbm2(rand, w, h, 6, 5, {
    warp: (u, v) => [u + (fbmWarp(warpRand, u, v, w, h) - 0.5) * 0.1, v],
  })
  const [canvas] = paintTexture(w, h, (_u, _v, x, y) => {
    const t = n[y * w + x]
    const a = Math.max(0, (t - 0.52) / 0.48)
    return [255, 255, 255, Math.round(a * 215)]
  })
  return toTexture(canvas)
}

export function marsTexture(w: number): THREE.CanvasTexture {
  const h = w / 2
  const rand = rng(51)
  const n = fbm2(rand, w, h, 6, 5)
  const [canvas] = paintTexture(w, h, (_u, v, x, y) => {
    const i = y * w + x
    const t = n[i]
    const lat = Math.abs(v - 0.5) * 2
    // 极冠
    if (lat > 0.92 + (n[i] - 0.5) * 0.08) return [245, 244, 240, 255]
    // 暗色玄武岩区域
    const dark = n[i] < 0.32
    const [r, g, b] = dark
      ? ramp(
          [
            [0, [120, 62, 44]],
            [0.5, [142, 74, 52]],
            [1, [158, 88, 58]],
          ],
          t,
        )
      : ramp(
          [
            [0, [168, 88, 52]],
            [0.45, [196, 112, 62]],
            [0.75, [216, 138, 82]],
            [1, [232, 164, 104]],
          ],
          t,
        )
    return [r, g, b, 255]
  })
  const ctx = canvas.getContext('2d')!
  drawCraters(ctx, rand, w, h, Math.round(w / 7), { maxR: 0.05, alpha: 0.4 })
  drawCraters(ctx, rand, w, h, Math.round(w / 4), { maxR: 0.018, alpha: 0.35 })
  return toTexture(canvas)
}

export function jupiterTexture(w: number): THREE.CanvasTexture {
  const h = w / 2
  const rand = rng(61)
  // 域扭曲制造起伏的云带
  const warp = fbm2(rand, w, h, 3, 3)
  const bands = fbm2(rand, w, h, 4, 2, {
    warp: (u, v) => {
      const x = Math.min(w - 1, Math.round(u * w))
      const y = Math.min(h - 1, Math.round(v * h))
      return [u + (warp[y * w + x] - 0.5) * 0.18, v]
    },
  })
  const turb = fbm2(rand, w, h, 5, 16)
  const palette: Stop[] = [
    [0, [168, 128, 96]],
    [0.2, [212, 180, 140]],
    [0.35, [240, 224, 200]],
    [0.5, [188, 140, 104]],
    [0.62, [226, 192, 158]],
    [0.78, [148, 96, 74]],
    [1, [232, 206, 172]],
  ]
  const [canvas] = paintTexture(w, h, (_u, _v, x, y) => {
    const i = y * w + x
    const t = bands[i] * 0.8 + turb[i] * 0.2
    const [r, g, b] = ramp(palette, t)
    return [r, g, b, 255]
  })
  const ctx = canvas.getContext('2d')!
  // 大红斑（位置固定，明显可见）
  drawOvalSpot(ctx, w * 0.7, h * 0.66, w * 0.11, h * 0.16, -0.12, [
    [0, 'rgba(206,86,50,0.95)'],
    [0.5, 'rgba(188,74,44,0.9)'],
    [0.75, 'rgba(240,200,170,0.75)'],
    [1, 'rgba(240,200,170,0)'],
  ])
  // 红斑周围白色涡流
  ctx.save()
  ctx.globalAlpha = 0.35
  ctx.strokeStyle = '#f6ead8'
  ctx.lineWidth = Math.max(1, w / 256)
  ctx.beginPath()
  ctx.ellipse(w * 0.7, h * 0.66, w * 0.155, h * 0.23, -0.12, 0.3, Math.PI * 1.6)
  ctx.stroke()
  ctx.restore()
  return toTexture(canvas)
}

export function saturnTexture(w: number): THREE.CanvasTexture {
  const h = w / 2
  const rand = rng(71)
  const warpRand = rng(72) // 同 earthCloudTexture：必须在 fbm2 外创建一次
  const n = fbm2(rand, w, h, 4, 3, {
    warp: (u, v) => [u + (fbmWarp(warpRand, u, v, w, h) - 0.5) * 0.05, v],
  })
  const [canvas] = paintTexture(w, h, (_u, _v, x, y) => {
    const t = n[y * w + x]
    const [r, g, b] = ramp(
      [
        [0, [176, 148, 106]],
        [0.3, [212, 190, 146]],
        [0.55, [232, 214, 172]],
        [0.8, [204, 178, 132]],
        [1, [226, 206, 160]],
      ],
      t,
    )
    return [r, g, b, 255]
  })
  return toTexture(canvas)
}

export function uranusTexture(w: number): THREE.CanvasTexture {
  const h = w / 2
  const rand = rng(81)
  const n = fbm2(rand, w, h, 3, 2)
  const [canvas] = paintTexture(w, h, (_u, v, x, y) => {
    const t = n[y * w + x] * 0.5 + v * 0.5
    const [r, g, b] = ramp(
      [
        [0, [128, 204, 220]],
        [0.5, [152, 220, 234]],
        [1, [186, 236, 244]],
      ],
      t,
    )
    return [r, g, b, 255]
  })
  return toTexture(canvas)
}

export function neptuneTexture(w: number): THREE.CanvasTexture {
  const h = w / 2
  const rand = rng(91)
  const warp = fbm2(rand, w, h, 3, 3)
  const n = fbm2(rand, w, h, 5, 3, {
    warp: (u, v) => {
      const x = Math.min(w - 1, Math.round(u * w))
      const y = Math.min(h - 1, Math.round(v * h))
      const d = warp[y * w + x]
      return [u + (d - 0.5) * 0.12, v + (d - 0.5) * 0.04]
    },
  })
  const [canvas] = paintTexture(w, h, (_u, v, x, y) => {
    const t = n[y * w + x]
    const band = Math.sin(v * Math.PI * 6 + t * 3) * 0.5 + 0.5
    const [r, g, b] = ramp(
      [
        [0, [38, 66, 176]],
        [0.4, [56, 96, 208]],
        [0.7, [92, 138, 232]],
        [1, [150, 190, 246]],
      ],
      t * 0.55 + band * 0.45,
    )
    return [r, g, b, 255]
  })
  const ctx = canvas.getContext('2d')!
  // 大暗斑
  drawOvalSpot(ctx, w * 0.56, h * 0.52, w * 0.09, h * 0.13, 0.1, [
    [0, 'rgba(18,30,110,0.85)'],
    [0.7, 'rgba(30,52,140,0.6)'],
    [1, 'rgba(30,52,140,0)'],
  ])
  // 白色高云
  ctx.save()
  ctx.globalAlpha = 0.5
  for (let i = 0; i < 5; i++) {
    const cx = rand() * w
    const cy = h * (0.15 + rand() * 0.7)
    const rw = w * (0.05 + rand() * 0.08)
    const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, rw)
    g.addColorStop(0, 'rgba(255,255,255,0.75)')
    g.addColorStop(1, 'rgba(255,255,255,0)')
    ctx.fillStyle = g
    ctx.save()
    ctx.translate(cx, cy)
    ctx.scale(1, 0.35)
    ctx.translate(-cx, -cy)
    ctx.beginPath()
    ctx.arc(cx, cy, rw, 0, Math.PI * 2)
    ctx.fill()
    ctx.restore()
  }
  ctx.restore()
  return toTexture(canvas)
}

export function moonTexture(w: number): THREE.CanvasTexture {
  const h = w / 2
  const rand = rng(101)
  const n = fbm2(rand, w, h, 5, 5)
  const [canvas] = paintTexture(w, h, (_u, _v, x, y) => {
    const t = n[y * w + x]
    const [r, g, b] = ramp(
      [
        [0, [128, 128, 132]],
        [0.5, [166, 166, 170]],
        [1, [204, 204, 208]],
      ],
      t,
    )
    return [r, g, b, 255]
  })
  const ctx = canvas.getContext('2d')!
  // 月海（暗色斑块）
  ctx.save()
  ctx.globalAlpha = 0.4
  for (let i = 0; i < 9; i++) {
    const cx = rand() * w
    const cy = h * (0.25 + rand() * 0.5)
    const rw = w * (0.05 + rand() * 0.09)
    const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, rw)
    g.addColorStop(0, 'rgba(88,88,94,0.9)')
    g.addColorStop(1, 'rgba(88,88,94,0)')
    ctx.fillStyle = g
    ctx.beginPath()
    ctx.ellipse(cx, cy, rw, rw * 0.8, rand() * 3, 0, Math.PI * 2)
    ctx.fill()
  }
  ctx.restore()
  drawCraters(ctx, rand, w, h, Math.round(w / 5), { maxR: 0.05, alpha: 0.5 })
  drawCraters(ctx, rand, w, h, Math.round(w / 2.5), { maxR: 0.016, alpha: 0.4 })
  return toTexture(canvas)
}

/**
 * 土星环：方形画布同心圆带（RingGeometry 的平面 UV 直接采样）。
 * 含卡西尼缝（内暗缝）与多层透明度。
 */
export function saturnRingTexture(w: number): THREE.CanvasTexture {
  const [canvas, ctx] = makeCanvas(w, w)
  const cx = w / 2
  const cy = w / 2
  const maxR = w * 0.48
  // 环带定义：[起始比例, 结束比例, 颜色, alpha]
  const bands: [number, number, string, number][] = [
    [0.3, 0.42, '#8a7358', 0.32], // C 环 内暗
    [0.42, 0.5, '#e8d9b0', 0.85], // B 环 亮
    [0.5, 0.58, '#b8a488', 0.75],
    [0.58, 0.63, '#f2e6c4', 0.9],
    [0.63, 0.66, '#000000', 0], // 卡西尼缝
    [0.66, 0.8, '#e2d2ac', 0.7], // A 环
    [0.8, 0.83, '#000000', 0], // 恩克缝
    [0.83, 0.92, '#d6c49a', 0.4],
  ]
  for (let y = 0; y < w; y++) {
    for (let x = 0; x < w; x++) {
      const dx = x - cx
      const dy = y - cy
      const r = Math.sqrt(dx * dx + dy * dy) / maxR
      if (r < 0.3 || r > 0.92) continue
      for (const [t0, t1, color, alpha] of bands) {
        if (r >= t0 && r < t1) {
          const edge = Math.min((r - t0) / (t1 - t0), 1, (t1 - r) / (t1 - t0)) * 2
          const a = alpha * Math.min(1, Math.max(0, edge * 2))
          if (a > 0.01) {
            ctx.fillStyle = color
            ctx.globalAlpha = a
            ctx.fillRect(x, y, 1, 1)
          }
          break
        }
      }
    }
  }
  ctx.globalAlpha = 1
  return toTexture(canvas)
}

/** 天王星淡环 */
export function uranusRingTexture(w: number): THREE.CanvasTexture {
  const [canvas, ctx] = makeCanvas(w, w)
  const cx = w / 2
  const maxR = w * 0.48
  const bands: [number, number, string, number][] = [
    [0.62, 0.78, '#bfe8f2', 0.5],
    [0.85, 0.9, '#9fd8e8', 0.28],
  ]
  for (let y = 0; y < w; y++) {
    for (let x = 0; x < w; x++) {
      const r = Math.sqrt((x - cx) ** 2 + (y - cx) ** 2) / maxR
      for (const [t0, t1, color, alpha] of bands) {
        if (r >= t0 && r < t1) {
          ctx.fillStyle = color
          ctx.globalAlpha = alpha
          ctx.fillRect(x, y, 1, 1)
          break
        }
      }
    }
  }
  ctx.globalAlpha = 1
  return toTexture(canvas)
}

/** 太阳光晕 / 星光：径向渐变 */
export function glowTexture(size = 256, inner = 'rgba(255,255,255,1)', mid = 'rgba(255,220,140,0.35)'): THREE.CanvasTexture {
  const [canvas, ctx] = makeCanvas(size, size)
  const g = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2)
  g.addColorStop(0, inner)
  g.addColorStop(0.25, mid)
  g.addColorStop(1, 'rgba(255,255,255,0)')
  ctx.fillStyle = g
  ctx.fillRect(0, 0, size, size)
  const tex = toTexture(canvas, false)
  return tex
}

/** 星云：柔和彩色噪点云团（径向衰减避免画布方形边缘） */
export function nebulaTexture(size = 256, r = 120, g = 90, b = 220): THREE.CanvasTexture {
  const [canvas, ctx] = makeCanvas(size, size)
  const rand = rng(1000 + r * 3 + g * 5 + b)
  const n = fbm2(rand, size, size, 5, 2)
  const img = ctx.createImageData(size, size)
  const half = size / 2
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const i = y * size + x
      const t = n[i]
      const a = Math.pow(Math.max(0, (t - 0.45) / 0.55), 2.2) * 160
      // 径向衰减：边缘 alpha 平滑归零
      const d = Math.sqrt((x - half) ** 2 + (y - half) ** 2) / half
      const falloff = Math.max(0, 1 - d * d * d)
      img.data[i * 4] = r
      img.data[i * 4 + 1] = g
      img.data[i * 4 + 2] = b
      img.data[i * 4 + 3] = Math.round(a * falloff)
    }
  }
  ctx.putImageData(img, 0, 0)
  return toTexture(canvas, false)
}
