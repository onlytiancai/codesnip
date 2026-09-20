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
import { loadPlayerState, type PlayerState } from '../storage/playerState'

export interface SceneCallbacks {
  onReady(backend: Backend, degraded: boolean): void
  onFatal(message: string): void
  onPlayerState(state: { position: THREE.Vector3; speedMode: SpeedMode }): void
  onChunkCount(count: number): void
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
  private flight!: FlightController
  private chunkMgr!: ChunkManager

  private rafId: number | null = null
  private lastFrame = 0
  private disposed = false
  private initialized = false
  private resizeObserver: ResizeObserver | null = null

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

    // Chunk 管理
    this.chunkMgr = new ChunkManager(this.playerGroup)

    // 飞行控制
    this.flight = new FlightController(this.camera, this.playerGroup, this.canvas, {
      onLock: () => {},
      onUnlock: () => {},
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
    this.chunkMgr.resync()

    this.initialized = true
    this.callbacks.onReady(this.backend, this.degraded)
    this.lastFrame = performance.now()
    this.lastPlayerSave = performance.now()
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

  dispose(): void {
    this.disposed = true
    if (this.rafId) cancelAnimationFrame(this.rafId)
    this.resizeObserver?.disconnect()
    this.chunkMgr?.dispose()
    this.flight?.dispose()
    if (this.starfield) this.scene.remove(this.starfield.group)
    if (this.renderer) this.renderer.dispose()
  }
}