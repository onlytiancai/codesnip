/**
 * 飞船飞行控制器：PointerLockControls + WASD + Space/Ctrl + Shift 加速。
 *
 * 飞船世界位置由 playerGroup.position 承载；camera 是 playerGroup 的子节点，
 * 始终保持在 playerGroup 局部 (0,0,0) 附近，Float32 精度不会抖。
 *
 * 控制：
 *   - Mouse：左键 click 进入 pointer lock → 鼠标控制 yaw/pitch
 *   - W / S：前进 / 倒退（沿镜头方向）
 *   - A / D：左 / 右平移
 *   - Space / Ctrl：上升 / 下降
 *   - Shift：加速 ×5
 */

import * as THREE from 'three/webgpu'
import { PointerLockControls } from 'three/addons/controls/PointerLockControls.js'

export type SpeedMode = 'normal' | 'boost'

export interface FlightEvents {
  onLock(): void
  onUnlock(): void
  onSpeedChange(s: SpeedMode): void
}

export class FlightController {
  readonly controls: PointerLockControls
  readonly playerGroup: THREE.Group
  private camera: THREE.PerspectiveCamera
  private events: FlightEvents
  private keys = new Set<string>()
  private speedMode: SpeedMode = 'normal'
  private tmpForward = new THREE.Vector3()
  private tmpRight = new THREE.Vector3()
  private tmpUp = new THREE.Vector3(0, 1, 0)
  private tmpDir = new THREE.Vector3()
  private boundKeyDown = (e: KeyboardEvent) => this.onKey(e, true)
  private boundKeyUp = (e: KeyboardEvent) => this.onKey(e, false)
  private boundLock = () => {
    this.events.onLock()
  }
  private boundUnlock = () => {
    this.events.onUnlock()
  }

  /** scene 单位 / 秒 */
  private static readonly BASE_SPEED = 60
  private static readonly BOOST_MULTIPLIER = 6

  constructor(
    camera: THREE.PerspectiveCamera,
    playerGroup: THREE.Group,
    domElement: HTMLElement,
    events: FlightEvents,
  ) {
    this.camera = camera
    this.playerGroup = playerGroup
    this.events = events
    this.controls = new PointerLockControls(camera, domElement)
    this.controls.addEventListener('lock', this.boundLock)
    this.controls.addEventListener('unlock', this.boundUnlock)
    window.addEventListener('keydown', this.boundKeyDown)
    window.addEventListener('keyup', this.boundKeyUp)
  }

  /** 鼠标点击进入 pointer lock（需要由外部 canvas click handler 触发） */
  requestLock(): void {
    if (!this.controls.isLocked) this.controls.lock()
  }

  private onKey(e: KeyboardEvent, down: boolean) {
    const code = e.code
    if (down) {
      // Shift 切加速
      if (code === 'ShiftLeft' || code === 'ShiftRight') {
        if (this.speedMode !== 'boost') {
          this.speedMode = 'boost'
          this.events.onSpeedChange('boost')
        }
      }
      this.keys.add(code)
    } else {
      if (code === 'ShiftLeft' || code === 'ShiftRight') {
        if (this.speedMode !== 'normal') {
          this.speedMode = 'normal'
          this.events.onSpeedChange('normal')
        }
      }
      this.keys.delete(code)
    }
  }

  /** 每帧调用 dt 秒。推进飞船位置。 */
  update(dt: number): void {
    const keys = this.keys
    const f = (keys.has('KeyW') || keys.has('ArrowUp') ? 1 : 0) -
      (keys.has('KeyS') || keys.has('ArrowDown') ? 1 : 0)
    const r = (keys.has('KeyD') || keys.has('ArrowRight') ? 1 : 0) -
      (keys.has('KeyA') || keys.has('ArrowLeft') ? 1 : 0)
    const u = (keys.has('Space') ? 1 : 0) - (keys.has('ControlLeft') || keys.has('ControlRight') ? 1 : 0)

    if (f === 0 && r === 0 && u === 0) return

    // 镜头方向（前/右/上）
    this.camera.getWorldDirection(this.tmpForward)
    this.tmpForward.normalize()
    this.tmpRight.copy(this.tmpForward).cross(this.tmpUp).normalize()
    // tmpUp 重新叉乘保持正交
    const up = new THREE.Vector3().copy(this.tmpRight).cross(this.tmpForward).normalize()

    const speed = FlightController.BASE_SPEED *
      (this.speedMode === 'boost' ? FlightController.BOOST_MULTIPLIER : 1) * dt

    this.tmpDir.set(0, 0, 0)
    this.tmpDir.addScaledVector(this.tmpForward, f)
    this.tmpDir.addScaledVector(this.tmpRight, r)
    this.tmpDir.addScaledVector(up, u)
    this.tmpDir.normalize().multiplyScalar(speed)

    this.playerGroup.position.add(this.tmpDir)
  }

  /** 强制设位置（用于从存档恢复 / 传送） */
  setPosition(x: number, y: number, z: number): void {
    this.playerGroup.position.set(x, y, z)
  }

  /** 读取当前世界坐标 */
  getPosition(out: THREE.Vector3): void {
    out.copy(this.playerGroup.position)
  }

  /** 读取 camera 当前四元数（用于持久化朝向） */
  getQuaternion(out: THREE.Quaternion): void {
    out.copy(this.camera.quaternion)
  }

  /** 恢复朝向 */
  setQuaternion(q: THREE.Quaternion): void {
    this.camera.quaternion.copy(q)
  }

  /** 速度模式 */
  getSpeedMode(): SpeedMode {
    return this.speedMode
  }

  dispose(): void {
    this.controls.removeEventListener('lock', this.boundLock)
    this.controls.removeEventListener('unlock', this.boundUnlock)
    this.controls.disconnect()
    window.removeEventListener('keydown', this.boundKeyDown)
    window.removeEventListener('keyup', this.boundKeyUp)
  }
}