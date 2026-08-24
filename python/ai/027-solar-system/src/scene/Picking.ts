/**
 * 点击拾取：区分「拖拽」与「点击」（位移阈值 6px），
 * 命中天体 → 回调；空白处 → 取消选中。
 */

import * as THREE from 'three/webgpu'
import type { BodyId } from '../data/planetData'

export interface PickTarget {
  mesh: THREE.Object3D
  id: BodyId
}

const CLICK_THRESHOLD_PX = 6

export class Picking {
  private raycaster = new THREE.Raycaster()
  private pointer = new THREE.Vector2()
  private targets: PickTarget[] = []
  private downPos: { x: number; y: number } | null = null
  private onPick: (id: BodyId | null) => void
  private el: HTMLElement
  private camera: THREE.PerspectiveCamera

  private onPointerDown = (e: PointerEvent) => {
    if (e.button !== 0) return
    this.downPos = { x: e.clientX, y: e.clientY }
  }

  private onPointerUp = (e: PointerEvent) => {
    if (e.button !== 0 || !this.downPos) return
    const dx = e.clientX - this.downPos.x
    const dy = e.clientY - this.downPos.y
    this.downPos = null
    if (Math.hypot(dx, dy) > CLICK_THRESHOLD_PX) return // 是拖拽
    const rect = this.el.getBoundingClientRect()
    this.pointer.set(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1)
    this.raycaster.setFromCamera(this.pointer, this.camera)
    const meshes = this.targets.map((t) => t.mesh)
    const hits = this.raycaster.intersectObjects(meshes, false)
    if (hits.length > 0) {
      const hit = hits[0]
      // 命中对象可能是子对象？这里网格都是直接目标，向上找第一个注册过的
      let obj: THREE.Object3D | null = hit.object
      while (obj) {
        const t = this.targets.find((x) => x.mesh === obj)
        if (t) {
          this.onPick(t.id)
          return
        }
        obj = obj.parent
      }
    }
    this.onPick(null)
  }

  constructor(el: HTMLElement, camera: THREE.PerspectiveCamera, onPick: (id: BodyId | null) => void) {
    this.el = el
    this.camera = camera
    this.onPick = onPick
    el.addEventListener('pointerdown', this.onPointerDown)
    el.addEventListener('pointerup', this.onPointerUp)
  }

  setTargets(targets: PickTarget[]): void {
    this.targets = targets
  }

  dispose(): void {
    this.el.removeEventListener('pointerdown', this.onPointerDown)
    this.el.removeEventListener('pointerup', this.onPointerUp)
  }
}
