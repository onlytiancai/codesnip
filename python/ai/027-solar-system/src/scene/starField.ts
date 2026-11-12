/**
 * 星空背景：InstancedMesh 小平面 + 大尺寸星云 Sprite。
 *
 * 为什么不用 Points：WebGPU 后端强制 1px 点（size/sizeAttenuation 无效），
 * InstancedMesh 在双后端都是单 draw call，尺寸/亮度完全可控。
 * 星星分布在球壳上并朝向原点（相机始终在壳内，星星始终正对视线）。
 */

import * as THREE from 'three/webgpu'
import { glowTexture, nebulaTexture } from './ProceduralTextures'
import type { QualityProfile } from '../utils/performance'

export interface StarFieldHandle {
  group: THREE.Group
  /** 缓慢旋转星空（制造微妙视差） */
  update(dt: number): void
}

const STAR_COLORS = [
  [255, 255, 255], // 白
  [255, 252, 244], // 暖白
  [224, 236, 255], // 冷白
  [255, 228, 196], // 橙黄
  [200, 220, 255], // 蓝
]

export function createStarField(profile: QualityProfile): StarFieldHandle {
  const group = new THREE.Group()
  const rand = mulberry(20260818)

  // ---- 星星 ----
  const starTex = glowTexture(64, 'rgba(255,255,255,1)', 'rgba(255,255,255,0.5)')
  const geo = new THREE.PlaneGeometry(1, 1)
  const mat = new THREE.MeshBasicMaterial({
    map: starTex,
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
  })
  const stars = new THREE.InstancedMesh(geo, mat, profile.starCount)
  stars.instanceMatrix.setUsage(THREE.StaticDrawUsage)

  const m = new THREE.Matrix4()
  const q = new THREE.Quaternion()
  const up = new THREE.Vector3(0, 1, 0)
  const pos = new THREE.Vector3()
  const color = new THREE.Color()
  const dir = new THREE.Vector3()
  const INNER = 220
  const OUTER = 430

  for (let i = 0; i < profile.starCount; i++) {
    // 球壳上均匀随机方向
    dir.set(rand() * 2 - 1, rand() * 2 - 1, rand() * 2 - 1)
    if (dir.lengthSq() < 0.01) dir.set(0, 1, 0)
    dir.normalize()
    pos.copy(dir).multiplyScalar(INNER + rand() * (OUTER - INNER))
    // 朝向原点
    q.setFromUnitVectors(up, dir.clone().negate())
    const s = 0.4 + rand() * 2.2
    m.compose(pos, q, new THREE.Vector3(s, s, 1))
    stars.setMatrixAt(i, m)
    const c = STAR_COLORS[Math.floor(rand() * STAR_COLORS.length)]
    color.setRGB(c[0] / 255, c[1] / 255, c[2] / 255)
    stars.setColorAt(i, color)
  }
  stars.instanceMatrix.needsUpdate = true
  stars.renderOrder = -10
  group.add(stars)

  // ---- 星云 ----
  const nebulaDefs = [
    { color: [140, 90, 220], size: 260, opacity: 0.16 },
    { color: [70, 110, 230], size: 200, opacity: 0.14 },
    { color: [60, 170, 190], size: 170, opacity: 0.12 },
    { color: [200, 110, 190], size: 150, opacity: 0.11 },
    { color: [90, 140, 250], size: 220, opacity: 0.1 },
  ]
  for (let i = 0; i < profile.nebulaCount; i++) {
    const def = nebulaDefs[i]
    const tex = nebulaTexture(256, def.color[0], def.color[1], def.color[2])
    const sprite = new THREE.Sprite(
      new THREE.SpriteMaterial({
        map: tex,
        transparent: true,
        opacity: def.opacity,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      }),
    )
    sprite.scale.setScalar(def.size)
    dir.set(rand() * 2 - 1, rand() * 0.8 - 0.2, rand() * 2 - 1).normalize()
    sprite.position.copy(dir).multiplyScalar(360 + rand() * 80)
    sprite.renderOrder = -9
    group.add(sprite)
  }

  return {
    group,
    update(dt: number) {
      group.rotation.y += dt * 0.004
    },
  }
}

function mulberry(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a |= 0
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}
