/**
 * 相机控制器：OrbitControls + flyTo 平滑聚焦 + 跟随模式。
 */

import * as THREE from 'three/webgpu'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { cancelTween, startTween } from '../utils/tween'
import type { BuiltBody } from './buildSolarSystem'

export interface FlyTarget {
  position: THREE.Vector3
  /** 目标点世界坐标（跟随锚点） */
  anchor: THREE.Object3D | null
  /** 相机到目标的距离 */
  distance: number
  /** 锁定仰角（弧度，>=0 时强制） */
  minElevation?: number
}

const OVERVIEW_POS = new THREE.Vector3(0, 32, 66)
const OVERVIEW_TARGET = new THREE.Vector3(0, 0, 0)

export class CameraController {
  readonly controls: OrbitControls
  private camera: THREE.PerspectiveCamera
  private followAnchor: THREE.Object3D | null = null
  private flyToken: symbol | null = null
  private flying = false

  constructor(camera: THREE.PerspectiveCamera, domElement: HTMLElement) {
    this.camera = camera
    this.controls = new OrbitControls(camera, domElement)
    this.controls.enableDamping = true
    this.controls.dampingFactor = 0.08
    this.controls.minDistance = 2
    this.controls.maxDistance = 200
    this.controls.autoRotateSpeed = 0.6
    this.controls.maxPolarAngle = Math.PI * 0.92
    this.controls.target.copy(OVERVIEW_TARGET)
  }

  /** 每帧调用（在 controls.update 之前） */
  preUpdate(): void {
    if (this.followAnchor && !this.flying) {
      this.followAnchor.getWorldPosition(this.controls.target)
    }
  }

  update(): void {
    this.controls.update()
  }

  /** 平滑飞向天体 */
  flyToBody(body: BuiltBody): void {
    const r = Math.max(0.5, body.mesh.geometry.boundingSphere?.radius ?? 1)
    const isSun = body.id === 'sun'
    // 稍远的聚焦距离：行星占画面约 1/4，纹理细节刚好清晰
    const distance = Math.max(isSun ? r * 3.8 : r * 5.4, 2.2)
    const target = new THREE.Vector3()
    body.pivot.getWorldPosition(target)
    this.flyTo(target, distance, body.pivot)
  }

  /** 平滑飞到指定位置 */
  flyTo(target: THREE.Vector3, distance: number, anchor: THREE.Object3D | null, minElevation = 0.35): void {
    cancelTween(this.flyToken)
    const fromPos = this.camera.position.clone()
    const fromTarget = this.controls.target.clone()

    // 保持当前方位角，只调整距离与俯仰
    const dir = fromPos.clone().sub(fromTarget)
    dir.normalize()
    dir.y = Math.max(dir.y, minElevation)
    dir.normalize()
    const toPos = target.clone().add(dir.multiplyScalar(distance))

    this.flying = true
    this.followAnchor = null
    this.flyToken = startTween(0, 1, 1400, (k) => {
      this.camera.position.lerpVectors(fromPos, toPos, k)
      this.controls.target.lerpVectors(fromTarget, target, k)
    }, { done: () => {
      this.flying = false
      this.followAnchor = anchor
      this.controls.target.copy(target)
    }})
  }

  /** 重置到太阳系全景 */
  resetView(): void {
    cancelTween(this.flyToken)
    const fromPos = this.camera.position.clone()
    const fromTarget = this.controls.target.clone()
    this.flying = true
    this.followAnchor = null
    this.flyToken = startTween(0, 1, 1300, (k) => {
      this.camera.position.lerpVectors(fromPos, OVERVIEW_POS, k)
      this.controls.target.lerpVectors(fromTarget, OVERVIEW_TARGET, k)
    }, { done: () => {
      this.flying = false
      this.controls.target.copy(OVERVIEW_TARGET)
    }})
  }

  /** 停止跟随（取消选中时） */
  stopFollow(): void {
    this.followAnchor = null
  }

  get isFollowing(): boolean {
    return this.followAnchor !== null
  }

  setAutoRotate(v: boolean): void {
    this.controls.autoRotate = v
  }
}
