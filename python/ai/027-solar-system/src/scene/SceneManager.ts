/**
 * SceneManager：3D 场景总编排。
 * 渲染循环 / 时间系统（公转、自转、调速、暂停）/ 选中高亮 / 显示模式 /
 * 设置项实时生效 / bloom / FPS 探测 / resize / dispose。
 */

import * as THREE from 'three/webgpu'
import { CSS2DRenderer } from 'three/addons/renderers/CSS2DRenderer.js'
import { createRenderer, type Backend, type RendererInfo } from './rendererFactory'
import { buildSolarSystem, type BuiltBody, type SolarSystemHandles } from './buildSolarSystem'
import { createStarField, type StarFieldHandle } from './starField'
import { CameraController } from './CameraController'
import { Picking } from './Picking'
import { detectInitialQuality, probeLowFps, qualityProfile, type QualityProfile } from '../utils/performance'
import { updateTweens } from '../utils/tween'
import type { BodyId } from '../data/planetData'
import type { AppSettings, SceneApi, ViewMode } from '../state/AppState'
import { playSelect } from '../utils/audio'

export interface SceneCallbacks {
  onReady(backend: Backend, degraded: boolean): void
  onSelect(id: BodyId | null): void
  onQualitySlow(): void
  onFatal(message: string): void
}

interface OrbitEntry {
  group: THREE.Object3D
  phase: number
  seconds: number
}

export class SceneManager implements SceneApi {
  private container: HTMLElement
  private canvas: HTMLCanvasElement
  private callbacks: SceneCallbacks

  private renderer!: THREE.WebGPURenderer
  private backend!: Backend
  private degraded = false
  private scene = new THREE.Scene()
  private camera!: THREE.PerspectiveCamera
  private cssRenderer!: CSS2DRenderer
  private cssElement!: HTMLElement
  private pipeline: THREE.RenderPipeline | null = null
  private bloomOn = true

  private handles!: SolarSystemHandles
  private starfield!: StarFieldHandle
  private cameraCtrl!: CameraController
  private picking!: Picking

  private profile!: QualityProfile
  private settings!: AppSettings
  private viewMode: ViewMode = 'explore'
  private timeScale = 1
  private paused = false
  private selectedId: BodyId | null = null

  private orbitTime = 0
  private spinTime = 0
  private orbitEntries: OrbitEntry[] = []
  private timer = new THREE.Timer()
  private fpsSamples: number[] = []
  private fpsFrames = 0
  private fpsLast = performance.now()
  private fpsVerdictDone = false
  private disposed = false
  private resizeObserver!: ResizeObserver

  constructor(container: HTMLElement, canvas: HTMLCanvasElement, callbacks: SceneCallbacks) {
    this.container = container
    this.canvas = canvas
    this.callbacks = callbacks
    this.settings = { quality: 'auto' } as AppSettings // init 后由 applySettings 覆盖
  }

  async init(initialSettings: AppSettings): Promise<void> {
    this.settings = initialSettings
    const resolved: 'high' | 'low' =
      initialSettings.quality === 'low' ? 'low' : initialSettings.quality === 'high' ? 'high' : detectInitialQuality()
    this.profile = qualityProfile(resolved)

    try {
      const info: RendererInfo = await createRenderer({
        canvas: this.canvas,
        dpr: Math.min(window.devicePixelRatio || 1, this.profile.dprCap),
        antialias: this.profile.antialias,
      })
      if (this.disposed) return // 初始化期间被卸载（如 React 树崩溃）
      this.renderer = info.renderer
      this.backend = info.backend
      this.degraded = info.degraded
    } catch (e) {
      this.callbacks.onFatal(e instanceof Error ? e.message : String(e))
      return
    }

    this.camera = new THREE.PerspectiveCamera(50, 1, 0.1, 3000)
    this.camera.position.set(0, 32, 66)

    // CSS2D 标签层（DOM overlay，指针穿透）
    this.cssRenderer = new CSS2DRenderer()
    this.cssElement = this.cssRenderer.domElement
    this.cssElement.classList.add('css2d-layer')
    this.container.appendChild(this.cssElement)

    // 场景内容
    this.handles = buildSolarSystem(this.scene, this.renderer, this.profile, (id) => this.selectBody(id, { fly: true }))
    this.starfield = createStarField(this.profile)
    this.scene.add(this.starfield.group)

    this.collectOrbitEntries()

    this.cameraCtrl = new CameraController(this.camera, this.renderer.domElement)
    this.picking = new Picking(this.renderer.domElement, this.camera, (id) => {
      if (id) {
        this.selectBody(id, { fly: true })
      } else {
        this.cameraCtrl.stopFollow()
        this.callbacks.onSelect(null)
      }
    })
    this.picking.setTargets(
      [...this.handles.bodies.values()].map((b) => ({ mesh: b.mesh, id: b.id })),
    )

    // bloom 完全跳过：bloom 渲染管线在某些驱动下会产生大模糊半径泛光，覆盖行星造成失焦
    // 太阳发光改为 MeshBasicMaterial toneMapped:false 实现，清晰锐利
    this.bloomOn = false
    void this.profile.bloomAllowed // 保留字段引用
    void initialSettings.bloomEnabled

    this.applySettings(initialSettings)
    this.setViewMode('explore')
    this.applyLabelContent()

    // resize
    this.resizeObserver = new ResizeObserver(() => this.onResize())
    this.resizeObserver.observe(this.container)
    this.onResize()

    // FPS 探测（3 秒后判定）
    probeLowFps(() => this.currentFps(), (slow) => {
      if (slow && !this.fpsVerdictDone) {
        this.fpsVerdictDone = true
        this.callbacks.onQualitySlow()
      }
    })

    // 先通知就绪，再启动循环：即使渲染循环异常，UI 也不会卡死在 Loading
    this.callbacks.onReady(this.backend, this.degraded)

    // 开发调试入口：控制台可访问 scene / camera / 天体
    if (import.meta.env.DEV) {
      ;(window as unknown as { __solar?: unknown }).__solar = {
        scene: this.scene,
        camera: this.camera,
        bodies: this.handles.bodies,
        manager: this,
      }
    }

    this.timer = new THREE.Timer()
    this.renderer.setAnimationLoop(() => this.frame())
  }

  // ---------- 渲染循环 ----------

  private frame(): void {
    if (this.disposed) return
    this.timer.update()
    const dt = Math.min(this.timer.getDelta(), 0.1)
    this.trackFps(dt)

    // 时间推进（暂停只冻结天体时间，不冻结相机）
    if (!this.paused) {
      if (this.settings.orbitEnabled) this.orbitTime += dt * this.timeScale
      if (this.settings.spinEnabled) this.spinTime += dt * this.timeScale
    }

    this.updateBodies()
    this.cameraCtrl.preUpdate()
    this.cameraCtrl.update()
    updateTweens()
    this.starfield.update(dt)
    this.pulseSunGlow()

    try {
      if (this.pipeline && this.bloomOn) {
        this.pipeline.render()
      } else {
        this.renderer.render(this.scene, this.camera)
      }
      this.cssRenderer.render(this.scene, this.camera)
    } catch (e) {
      // 渲染异常兜底：bloom 管线失败则永久关闭，退回普通渲染；普通渲染也失败则停止循环
      if (this.pipeline && this.bloomOn) {
        console.warn('[renderer] bloom 渲染失败，已关闭（光晕 Sprite 兜底）：', e)
        this.pipeline = null
        this.bloomOn = false
      } else {
        console.error('[renderer] 渲染循环异常，已停止：', e)
        this.renderer.setAnimationLoop(null)
      }
    }
  }

  private updateBodies(): void {
    for (const { group, phase, seconds } of this.orbitEntries) {
      group.rotation.y = phase + (seconds > 0 ? (this.orbitTime / seconds) * Math.PI * 2 : 0)
    }
    for (const body of this.handles.bodies.values()) {
      if (body.spinSeconds !== 0) {
        body.mesh.rotation.y = (this.spinTime / body.spinSeconds) * Math.PI * 2
      }
      if (body.cloudMesh) {
        body.cloudMesh.rotation.y = body.mesh.rotation.y * 1.12 + 0.3
      }
    }
  }

  private pulseSunGlow(): void {
    // 太阳光晕已改用 bloom + emissive 自然渲染，不再做 sprite 脉动
    void this.handles?.sunGlow.length // 保留方法签名兼容
  }

  // ---------- FPS 探测 ----------

  private currentFps(): number {
    return this.fpsSamples.length ? this.fpsSamples[this.fpsSamples.length - 1] : 60
  }

  private trackFps(_dt: number): void {
    this.fpsFrames++
    const now = performance.now()
    if (now - this.fpsLast >= 500) {
      const fps = this.fpsFrames / ((now - this.fpsLast) / 1000)
      this.fpsSamples.push(fps)
      if (this.fpsSamples.length > 12) this.fpsSamples.shift()
      this.fpsFrames = 0
      this.fpsLast = now
    }
  }

  // ---------- 选中与高亮 ----------

  selectBody(id: BodyId | null, opts?: { fly?: boolean }): void {
    if (!this.handles || !this.cameraCtrl) return // init 未完成
    if (id === this.selectedId) {
      if (id && opts?.fly) this.cameraCtrl.flyToBody(this.handles.bodies.get(id)!)
      return
    }
    // 还原旧选中
    if (this.selectedId) this.setHighlight(this.handles.bodies.get(this.selectedId)!, false)
    this.selectedId = id
    if (id) {
      const body = this.handles.bodies.get(id)!
      this.setHighlight(body, true)
      if (opts?.fly) this.cameraCtrl.flyToBody(body)
      playSelect()
    } else {
      this.cameraCtrl.stopFollow()
    }
    this.callbacks.onSelect(id)
  }

  private setHighlight(body: BuiltBody, on: boolean): void {
    if (body.id !== 'sun') {
      const mat = body.mesh.material as THREE.MeshStandardMaterial
      if (on) {
        // 选中态用 outline 替代 emissive：避免触发 bloom
        mat.emissive.setHex(0x000000)
        mat.emissiveIntensity = 0
      } else {
        mat.emissive.setHex(0x000000)
        mat.emissiveIntensity = 0
      }
    }
    if (body.orbitLine) {
      const m = body.orbitLine.material as THREE.LineBasicMaterial
      m.color.set(on ? body.data.accent : 0x7f96c8)
      m.opacity = on ? 0.9 : this.orbitBaseOpacity()
    }
    body.label?.element.classList.toggle('active', on)
  }

  private orbitBaseOpacity(): number {
    return this.viewMode === 'orbits' ? 0.8 : 0.32
  }

  // ---------- SceneApi 实现 ----------

  resetView(): void {
    if (!this.cameraCtrl) return
    this.cameraCtrl.resetView()
  }

  setTimeScale(v: number): void {
    this.timeScale = v
  }

  setPaused(p: boolean): void {
    this.paused = p
  }

  setResolvedQuality(q: 'high' | 'low'): void {
    if (!this.renderer) return // init 尚未完成（init 内部自行按探测结果配置）
    const profile = qualityProfile(q)
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, profile.dprCap))
    if (q === 'low') this.bloomOn = false
    this.onResize()
  }

  setViewMode(mode: ViewMode): void {
    if (!this.handles) return
    this.viewMode = mode
    this.applyLabelVisibility()
    this.applyLabelContent()
    // 轨道模式：行星变暗、轨道线突出
    const dim = mode === 'orbits'
    for (const body of this.handles.bodies.values()) {
      if (body.id === 'sun') continue
      for (const mat of body.dimmable) {
        if (mat instanceof THREE.MeshStandardMaterial || mat instanceof THREE.MeshPhongMaterial || mat instanceof THREE.MeshLambertMaterial) {
          mat.color.setScalar(dim ? 0.42 : 1)
        }
      }
    }
    for (const line of this.handles.orbitLines) {
      const m = line.material as THREE.LineBasicMaterial
      if (this.selectedId) continue // 选中轨道保持高亮
      m.opacity = this.orbitBaseOpacity()
      m.color.set(dim ? 0x9db8ff : 0x7f96c8)
    }
    // 科普模式：强制显示标签与数据
    this.applySettings(this.settings)
  }

  applySettings(s: AppSettings): void {
    this.settings = s
    if (!this.handles || !this.cameraCtrl) return // init 未完成，init 内部会应用设置
    // 轨道线
    for (const line of this.handles.orbitLines) line.visible = s.showOrbits
    for (const body of this.handles.bodies.values()) {
      if (body.moonOrbitLine) body.moonOrbitLine.visible = s.showOrbits
    }
    // 星空
    this.starfield.group.visible = s.showStarfield
    // 辅助线
    this.handles.guideGroup.visible = s.showGuide
    // 标签
    this.applyLabelVisibility()
    this.applyLabelContent()
    // 自动旋转
    this.cameraCtrl.setAutoRotate(s.autoRotate)
    // 辉光
    this.bloomOn = s.bloomEnabled && this.profile.bloomAllowed && this.settings.quality !== 'low'
    // 阴影
    this.handles.sunLight.castShadow = s.shadowsEnabled
    for (const body of this.handles.bodies.values()) {
      body.mesh.castShadow = s.shadowsEnabled
      body.mesh.receiveShadow = s.shadowsEnabled
    }
    // 质量（运行时只调 DPR / 辉光等低成本项）
    if (s.quality === 'low') this.setResolvedQuality('low')
    else if (s.quality === 'high') this.setResolvedQuality('high')
  }

  private applyLabelVisibility(): void {
    const show = this.settings.showLabels || this.viewMode === 'science'
    for (const label of this.handles.labels) label.visible = show
    this.applyLabelContent()
  }

  private applyLabelContent(): void {
    const dataOn = this.settings.showData || this.viewMode === 'science'
    const distOn = this.settings.showDistances || this.viewMode === 'science'
    const periodOn = this.settings.showPeriods || this.viewMode === 'science'
    for (const label of this.handles.labels) {
      const distEl = label.userData.subDistEl as HTMLElement | undefined
      const periodEl = label.userData.subPeriodEl as HTMLElement | undefined
      if (distEl) distEl.hidden = !(dataOn && distOn)
      if (periodEl) periodEl.hidden = !(dataOn && periodOn)
      label.element.classList.toggle('with-sub', dataOn && (distOn || periodOn))
    }
  }

  // ---------- 其他 ----------

  private collectOrbitEntries(): void {
    this.orbitEntries = []
    const moon = this.handles.bodies.get('moon')
    for (const body of this.handles.bodies.values()) {
      if (body.orbitGroup && body.orbitSeconds > 0) {
        this.orbitEntries.push({
          group: body.orbitGroup,
          phase: body.orbitGroup.userData.phase ?? 0,
          seconds: body.orbitSeconds,
        })
      }
      if (body.moonOrbitGroup && moon) {
        this.orbitEntries.push({
          group: body.moonOrbitGroup,
          phase: 0,
          seconds: moon.orbitSeconds,
        })
      }
    }
  }

  private onResize(): void {
    const w = this.container.clientWidth || window.innerWidth
    const h = this.container.clientHeight || window.innerHeight
    this.camera.aspect = w / h
    this.camera.updateProjectionMatrix()
    this.renderer.setSize(w, h, false)
    this.cssRenderer.setSize(w, h)
  }

  private async initBloom(): Promise<void> {
    if (!this.profile.bloomAllowed) return
    try {
      const tsl = await import('three/tsl')
      const { bloom } = await import('three/addons/tsl/display/BloomNode.js')
      this.pipeline = new THREE.RenderPipeline(this.renderer)
      // 极高阈值 1.0 + 弱强度：只有自发光 sprite/太阳本体（toneMapped:false 会跳过色彩管理）超过阈值
      // 普通行星 8-bit 颜色上限是 1.0，永远不会触发泛光
      this.pipeline.outputNode = bloom(tsl.pass(this.scene, this.camera).getTextureNode(), 1.0, 0.6, 0.1)
      console.info('[renderer] bloom 已启用（RenderPipeline）')
    } catch (e) {
      console.warn('[renderer] bloom 不可用，已跳过（太阳光晕兜底）：', e)
      this.pipeline = null
    }
  }

  dispose(): void {
    this.disposed = true
    this.renderer?.setAnimationLoop(null)
    this.picking?.dispose()
    this.resizeObserver?.disconnect()
    this.cssElement?.remove()
    this.renderer?.dispose()
  }
}
