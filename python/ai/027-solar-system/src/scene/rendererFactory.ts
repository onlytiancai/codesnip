/**
 * 渲染器工厂：WebGPU 优先，自动降级 WebGL2。
 *
 * 降级链：
 * 1. URL 参数 ?forceWebGL=1 或无 navigator.gpu → 直接 WebGL2 后端（forceWebGL: true）
 * 2. await renderer.init() 抛错 → dispose 后以 forceWebGL 重建
 * 3. 首帧 render() 抛错（WebGPU 驱动缺陷）→ dispose 后以 forceWebGL 重建
 * 4. 两个后端都失败 → 抛出，由上层显示友好错误页
 *
 * 注意：r185 的 WebGPURenderer 内置 WebGLBackend 降级；
 * 经典 WebGLRenderer 只存在于主构建，本项目统一走 three/webgpu 单一构建。
 */

import * as THREE from 'three/webgpu'

export type Backend = 'webgpu' | 'webgl2'

export interface RendererInfo {
  renderer: THREE.WebGPURenderer
  backend: Backend
  /** 是否发生了降级（用于 UI 提示） */
  degraded: boolean
}

interface CreateOptions {
  canvas: HTMLCanvasElement
  dpr: number
  antialias: boolean
}

async function tryCreate(opts: CreateOptions, forceWebGL: boolean): Promise<THREE.WebGPURenderer> {
  const renderer = new THREE.WebGPURenderer({
    canvas: opts.canvas,
    antialias: opts.antialias,
    forceWebGL,
  })
  renderer.setPixelRatio(opts.dpr)
  renderer.setSize(opts.canvas.clientWidth || window.innerWidth, opts.canvas.clientHeight || window.innerHeight, false)
  await renderer.init()
  return renderer
}

/** 首帧防线：WebGPU 驱动可能 init 成功但 render 即崩 */
function firstRenderOk(renderer: THREE.WebGPURenderer): boolean {
  const probeScene = new THREE.Scene()
  const probeCam = new THREE.PerspectiveCamera()
  try {
    renderer.render(probeScene, probeCam)
    return true
  } catch (e) {
    console.warn('[renderer] 首帧渲染失败，回退 WebGL：', e)
    return false
  }
}

/** r185 运行时存在 isWebGPUBackend 标记，但 @types 未收录，做窄化断言 */
function isWebGPU(renderer: THREE.WebGPURenderer): boolean {
  return (renderer.backend as unknown as { isWebGPUBackend?: boolean }).isWebGPUBackend === true
}

export async function createRenderer(opts: CreateOptions): Promise<RendererInfo> {
  const forceWebGL = new URLSearchParams(location.search).has('forceWebGL') || !('gpu' in navigator)
  const desired: Backend = forceWebGL ? 'webgl2' : 'webgpu'

  let renderer: THREE.WebGPURenderer | null = null
  try {
    renderer = await tryCreate(opts, forceWebGL)
  } catch (e) {
    console.warn('[renderer] 初始化失败，尝试 WebGL 兼容模式：', e)
    renderer?.dispose()
    renderer = null
  }

  if (!renderer) {
    try {
      renderer = await tryCreate(opts, true) // 强制 WebGL2 重建
    } catch (e) {
      console.error('[renderer] WebGL 也不可用：', e)
      throw new Error('当前浏览器不支持 WebGPU 和 WebGL，无法显示 3D 场景')
    }
  }

  if (desired === 'webgpu' && !isWebGPU(renderer)) {
    // 构造时发生内置降级或首帧失败，统一走首帧防线再确认一次
    if (!firstRenderOk(renderer)) {
      renderer.dispose()
      renderer = await tryCreate(opts, true)
    }
  }

  const backend: Backend = isWebGPU(renderer) ? 'webgpu' : 'webgl2'
  const degraded = desired === 'webgpu' && backend !== 'webgpu'

  // 统一的色调映射，双后端一致
  renderer.toneMapping = THREE.ACESFilmicToneMapping
  renderer.toneMappingExposure = 1.1

  console.info(`[renderer] 后端：${backend === 'webgpu' ? 'WebGPU' : 'WebGL2 兼容模式'}${degraded ? '（已降级）' : ''}`)
  return { renderer, backend, degraded }
}
