/**
 * SceneManager（第一人称自由飞行版）。
 *
 * - playerGroup：摄像机世界锚点
 *   - camera：FPS 视角，PointerLockControls 控制朝向
 *   - ChunkManager 的所有 chunk
 * - starfield：远景星空（挂在 scene，跟随相机旋转）
 * - FlightController：WASD + Shift 加速
 *
 * init 流程：
 *   1. renderer 初始化
 *   2. 创建 scene / camera / 灯光 / starfield
 *   3. 创建 playerGroup + 挂 camera
 *   4. 创建 ChunkManager + FlightController
 *   5. 从 IndexedDB 读 playerState → 恢复 playerGroup.position + camera.quaternion
 *   6. 启动 RAF
 */

import * as THREE from 'three/webgpu'
import { createRenderer, type Backend, type RendererInfo } from './rendererFactory'
import { createStarField, type StarFieldHandle } from './StarField'
import { FlightController, type SpeedMode } from './FlightController'
import { ChunkManager } from './ChunkManager'
import { detectInitialQuality, qualityProfile, type QualityProfile as QualityProfileType } from '../utils/performance'
import { loadPlayerState } from '../storage/playerState'

export type EscapeState = 'playing' | 'paused' | 'menu'

export interface SceneCallbacks {
  onReady(backend: Backend, degraded: boolean): void
  onFatal(message: string): void
  onPlayerState(state: { position: THREE.Vector3; speedMode: SpeedMode }): void
  onChunkCount(count: number): void
  onEscapeState(state: EscapeState): void
}

export class SceneManager {
  private container: HTMLElement
  private canvas: HTMLCanvasElement
  private callbacks: SceneCallbacks

  private renderer!: THREE.WebGPURenderer
  private backend!: Backend
  private degraded = false
  private scene = new THREE.Scene()
  private camera!: THREE.PerspectiveCamera
  private profile!: QualityProfileType

  private playerGroup!: THREE.Group
  private starfield!: StarFieldHandle
  /** 公开以便 UniverseCanvas 调用 setPosition / getQuaternion / requestLock */
  flight!: FlightController
  private chunkMgr!: ChunkManager

  private rafId: number | null = null
  private lastFrame = 0
  private disposed = false
  private initialized = false
  private resizeObserver: ResizeObserver | null = null

  private escapeState: EscapeState = 'menu'

  /** 持久化 player state 的定时器 */
  private lastPlayerSave = 0

  constructor(container: HTMLElement, canvas: HTMLCanvasElement, callbacks: SceneCallbacks) {
    this.container = container
    this.canvas = canvas
    this.callbacks = callbacks
  }

  async init(): Promise<void> {
    const resolved: 'high' | 'low' = detectInitialQuality()
    this.profile = qualityProfile(resolved)

    try {
      const info: RendererInfo = await createRenderer({
        canvas: this.canvas,
        dpr: Math.min(window.devicePixelRatio || 1, this.profile.dprCap),
        antialias: this.profile.antialias,
      })
      if (this.disposed) return
      this.renderer = info.renderer
      this.backend = info.backend
      this.degraded = info.degraded
    } catch (e) {
      this.callbacks.onFatal(e instanceof Error ? e.message : String(e))
      return
    }

    this.camera = new THREE.PerspectiveCamera(72, 1, 0.5, 8000)
    this.playerGroup = new THREE.Group()
    this.playerGroup.name = 'player'
    this.scene.add(this.playerGroup)
    this.playerGroup.add(this.camera)
    this.camera.position.set(0, 0, 0)

    // 环境光（行星需要）
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.45))
    const sun = new THREE.DirectionalLight(0xfff4d6, 1.4)
    sun.position.set(50, 30, 50)
    this.scene.add(sun)

    // 星空
    this.starfield = createStarField(this.profile)
    this.scene.add(this.starfield.group)

    // Esc 状态机 + pointerlockchange 监听
    window.addEventListener('keydown', this.onKeyDown)
    document.addEventListener('pointerlockchange', this.onPointerLockChange)

    // Chunk 管理
    this.chunkMgr = new ChunkManager(this.playerGroup)

    // 飞行控制
    this.flight = new FlightController(this.camera, this.playerGroup, this.canvas, {
      onLock: () => this.setEscapeState('playing'),
      onUnlock: () => {
        // 如果用户主动 Esc（已经是 paused），保持 paused；否则进入 menu
        if (this.escapeState === 'playing') this.setEscapeState('menu')
      },
      onSpeedChange: () => {},
    })

    // Resize
    this.resizeObserver = new ResizeObserver(() => this.handleResize())
    this.resizeObserver.observe(this.container)
    this.handleResize()

    // 从 IndexedDB 恢复 player state
    try {
      const saved = await loadPlayerState()
      if (saved) {
        this.flight.setPosition(saved.position.x, saved.position.y, saved.position.z)
        if (saved.quaternion) {
          this.flight.setQuaternion(
            new THREE.Quaternion(
              saved.quaternion.x,
              saved.quaternion.y,
              saved.quaternion.z,
              saved.quaternion.w,
            ),
          )
        }
      }
    } catch {
      /* 忽略 */
    }

    // 触发 ChunkManager 立刻加载飞船周围 chunk
    try {
      this.chunkMgr.resync()
    } catch (e) {
      console.error('[SceneManager] initial chunk resync failed:', e)
    }

    this.initialized = true
    this.callbacks.onReady(this.backend, this.degraded)
    this.lastFrame = performance.now()
    this.lastPlayerSave = performance.now()
    this.callbacks.onEscapeState(this.escapeState)
    this.callbacks.onPlayerState({
      position: this.playerGroup.position.clone(),
      speedMode: this.flight.getSpeedMode(),
    })
    this.callbacks.onChunkCount(this.chunkMgr.getActiveCount())
    this.rafId = requestAnimationFrame(this.loop)
  }

  private handleResize = (): void => {
    const w = this.container.clientWidth || window.innerWidth
    const h = this.container.clientHeight || window.innerHeight
    this.renderer.setSize(w, h, false)
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, this.profile.dprCap))
    this.camera.aspect = w / h
    this.camera.updateProjectionMatrix()
  }

  private setEscapeState(s: EscapeState): void {
    this.escapeState = s
    this.callbacks.onEscapeState(s)
  }

  /** Esc 状态机：
   *  playing → 按 Esc → setEscapeState('paused') + exitPointerLock()
   *  paused  → 按 Esc → setEscapeState('menu')
   *  menu    → 不响应 Esc（用户需点击 canvas 重新 lock）
   */
  private onKeyDown = (e: KeyboardEvent): void => {
    if (e.code !== 'Escape') return
    if (this.disposed) return
    if (this.escapeState === 'playing') {
      this.setEscapeState('paused')
      document.exitPointerLock()
    } else if (this.escapeState === 'paused') {
      this.setEscapeState('menu')
    }
  }

  /** pointerlockchange：unlock 不直接切 state，由 onKeyDown 控制 paused → menu。
   *  只有真正从 menu/paused 回到 playing（点击 canvas）才在这里切。
   */
  private onPointerLockChange = (): void => {
    const locked = document.pointerLockElement === this.canvas
    if (locked) {
      this.setEscapeState('playing')
    }
    // unlock 不处理：onKeyDown 已经处理 paused/menu 切换；
    // 异常 unlock（如浏览器焦点丢失）会等到 onKeyDown 之后再决定。
    // 如果 state 还是 'playing' 且 unlocked（罕见：onKeyDown 没收到），
    // 下一个 RAF 自动修复为 'menu'（避免卡在 playing 但无 lock）
    if (!locked && this.escapeState === 'playing') {
      // 用 setTimeout 推迟到下一个 macrotask，等可能的 keydown handler 先执行
      setTimeout(() => {
        if (this.escapeState === 'playing') this.setEscapeState('menu')
      }, 0)
    }
  }

  /** 给 React 端调：从 menu 状态请求 lock */
  requestPointerLock(): void {
    if (this.escapeState === 'menu' || this.escapeState === 'paused') {
      void this.canvas.requestPointerLock()
    }
  }

  private loop = (): void => {
    if (this.disposed) return
    const now = performance.now()
    const dt = Math.min(0.1, (now - this.lastFrame) / 1000)
    this.lastFrame = now

    this.flight.update(dt)
    this.chunkMgr.update(dt, now / 1000)
    this.starfield.update(dt)

    this.renderer.render(this.scene, this.camera)

    // 每 200ms 通知 React 位置 + 速度
    if (now - this.lastPlayerSave > 200) {
      this.lastPlayerSave = now
      this.callbacks.onPlayerState({
        position: this.playerGroup.position.clone(),
        speedMode: this.flight.getSpeedMode(),
      })
      this.callbacks.onChunkCount(this.chunkMgr.getActiveCount())
    }

    this.rafId = requestAnimationFrame(this.loop)
  }

  /** 获取摄像机当前位置（React 用） */
  getPlayerPosition(out: THREE.Vector3): void {
    out.copy(this.playerGroup.position)
  }

  /** 传送到指定世界坐标 */
  teleport(x: number, y: number, z: number): void {
    if (!this.initialized) return
    this.flight.setPosition(x, y, z)
    this.chunkMgr.resync()
  }

  /** 取附近星系（距离 + 世界坐标 + id），按距离升序 */
  getNearbySystems(maxRadius: number): Array<{ id: number; worldPos: THREE.Vector3; distance: number }> {
    if (!this.initialized) return []
    const out: Array<{ id: number; worldPos: THREE.Vector3; distance: number }> = []
    const px = this.playerGroup.position.x
    const py = this.playerGroup.position.y
    const pz = this.playerGroup.position.z
    const stars = this.chunkMgr.getAllStars()
    for (const s of stars) {
      const dx = s.worldPos.x - px
      const dy = s.worldPos.y - py
      const dz = s.worldPos.z - pz
      const dist = Math.hypot(dx, dy, dz)
      if (dist <= maxRadius) {
        out.push({ id: s.id, worldPos: s.worldPos, distance: dist })
      }
    }
    out.sort((a, b) => a.distance - b.distance)
    return out
  }

  /** 投影 worldPos 到屏幕坐标（返回 NDC 与 canvas-px） */
  projectToScreen(worldPos: THREE.Vector3): { x: number; y: number; inFront: boolean } {
    const v = worldPos.clone().project(this.camera)
    const w = this.canvas.clientWidth
    const h = this.canvas.clientHeight
    return {
      x: (v.x * 0.5 + 0.5) * w,
      y: (-v.y * 0.5 + 0.5) * h,
      inFront: v.z < 1,
    }
  }

  dispose(): void {
    this.disposed = true
    if (this.rafId) cancelAnimationFrame(this.rafId)
    this.resizeObserver?.disconnect()
    window.removeEventListener('keydown', this.onKeyDown)
    document.removeEventListener('pointerlockchange', this.onPointerLockChange)
    this.chunkMgr?.dispose()
    this.flight?.dispose()
    if (this.starfield) this.scene.remove(this.starfield.group)
    if (this.renderer) this.renderer.dispose()
  }
}