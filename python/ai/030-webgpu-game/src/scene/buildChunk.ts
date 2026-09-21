/**
 * 从 ChunkData 构建 Three.js Mesh 树（恒星 + 行星 + 彗星 + 轨道线）。
 *
 * 返回 ChunkHandle：group（相对 chunk 锚点）+ 每帧 update + dispose。
 *
 * 性能：所有材质共享（无 GLSL 重编译）；geometry 复用。
 */

import * as THREE from 'three/webgpu'
import {
  earthTexture,
  glowTexture,
  jupiterTexture,
  marsTexture,
  mercuryTexture,
  neptuneTexture,
  sunTexture,
  uranusTexture,
  venusTexture,
} from './ProceduralTextures'
import { solveKepler } from '../universe/generateChunk'
import { type PlanetType } from '../universe/starClasses'
import type { ChunkData } from '../universe/generateChunk'

const SPHERE_GEO = new THREE.SphereGeometry(1, 32, 24)
const COMET_NUCLEUS_GEO = new THREE.SphereGeometry(1, 12, 8)

function textureFor(type: PlanetType): THREE.CanvasTexture {
  switch (type) {
    case 'lava':
      return mercuryTexture(256)
    case 'rocky':
      return marsTexture(256)
    case 'water':
      return earthTexture(256)
    case 'desert':
      return venusTexture(256)
    case 'ice':
      return uranusTexture(256)
    case 'gasGiant':
      return jupiterTexture(256)
    case 'iceGiant':
      return neptuneTexture(256)
    case 'dwarf':
      return marsTexture(128)
  }
}

function buildOrbitLine(radius: number, color = 0x3a4a78): THREE.Line {
  const pts: THREE.Vector3[] = []
  const seg = 64
  for (let i = 0; i <= seg; i++) {
    const a = (i / seg) * Math.PI * 2
    pts.push(new THREE.Vector3(Math.cos(a) * radius, 0, Math.sin(a) * radius))
  }
  const geo = new THREE.BufferGeometry().setFromPoints(pts)
  const mat = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.3 })
  return new THREE.Line(geo, mat)
}

export interface ChunkHandle {
  group: THREE.Group
  /** 每个恒星相对 chunk 锚点的局部坐标（[x, y, z] tuple），用于标签投影 */
  starsLocal: Array<[number, number, number]>
  update(dt: number, time: number): void
  dispose(): void
}

/** 复用材质（全局 lazy 缓存） */
const STAR_MATERIAL_CACHE = new Map<string, THREE.MeshBasicMaterial>()
function starMaterial(color: [number, number, number]): THREE.MeshBasicMaterial {
  const key = color.join(',')
  let mat = STAR_MATERIAL_CACHE.get(key)
  if (!mat) {
    const tex = sunTexture(256)
    tex.colorSpace = THREE.SRGBColorSpace
    mat = new THREE.MeshBasicMaterial({
      map: tex,
      color: new THREE.Color(color[0] / 255, color[1] / 255, color[2] / 255),
      toneMapped: false,
    })
    STAR_MATERIAL_CACHE.set(key, mat)
  }
  return mat
}

const HALO_MATERIAL_CACHE = new Map<string, THREE.SpriteMaterial>()
function haloMaterial(color: [number, number, number]): THREE.SpriteMaterial {
  const key = color.join(',')
  let mat = HALO_MATERIAL_CACHE.get(key)
  if (!mat) {
    const halo = glowTexture(
      64,
      `rgba(${color[0]},${color[1]},${color[2]},0.8)`,
      `rgba(${color[0]},${color[1]},${color[2]},0)`,
    )
    mat = new THREE.SpriteMaterial({
      map: halo,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      toneMapped: false,
    })
    HALO_MATERIAL_CACHE.set(key, mat)
  }
  return mat
}

const PLANET_MAT_CACHE = new Map<PlanetType, THREE.MeshStandardMaterial>()
function planetMaterial(type: PlanetType): THREE.MeshStandardMaterial {
  let mat = PLANET_MAT_CACHE.get(type)
  if (!mat) {
    const tex = textureFor(type)
    tex.colorSpace = THREE.SRGBColorSpace
    mat = new THREE.MeshStandardMaterial({ map: tex, roughness: 0.9, metalness: 0.05 })
    PLANET_MAT_CACHE.set(type, mat)
  }
  return mat
}

const COMET_TAIL_MAT = new THREE.LineBasicMaterial({
  color: 0xb5d6ff,
  transparent: true,
  opacity: 0.7,
  blending: THREE.AdditiveBlending,
  depthWrite: false,
})

export function buildChunk(data: ChunkData): ChunkHandle {
  const group = new THREE.Group()
  group.name = `chunk-${data.cx}-${data.cy}-${data.cz}`

  const starsLocal: Array<[number, number, number]> = data.stars.map((s) => [s.position[0], s.position[1], s.position[2]])

  // ---------- 恒星 ----------
  const starGroups: THREE.Group[] = []
  for (const star of data.stars) {
    const g = new THREE.Group()
    g.position.set(...star.position)
    g.name = `star`
    const mesh = new THREE.Mesh(SPHERE_GEO, starMaterial(star.class.color))
    mesh.scale.setScalar(star.radius)
    g.add(mesh)

    const halo = new THREE.Sprite(haloMaterial(star.class.color))
    halo.scale.setScalar(star.radius * 4)
    g.add(halo)
    group.add(g)
    starGroups.push(g)
  }

  // ---------- 行星 / 轨道线 / 彗星（每个恒星一子树） ----------
  const planetEntries: {
    orbitGroup: THREE.Group
    mesh: THREE.Mesh
    orbitPeriod: number
    spinPeriod: number
  }[] = []
  const cometEntries: {
    comet: ChunkData['comets'][0]
    node: THREE.Group
    tailPositions: Float32Array
    tailLine: THREE.Line
    starPos: THREE.Vector3
  }[] = []

  for (let si = 0; si < data.stars.length; si++) {
    const star = data.stars[si]
    const starPos = new THREE.Vector3(...star.position)

    const starGroup = new THREE.Group()
    starGroup.position.copy(starPos)
    starGroup.name = `system-${si}`
    group.add(starGroup)

    // 该恒星的行星
    const hostPlanets = data.planets.filter((p) => p.hostStarIndex === si)
    for (let pi = 0; pi < hostPlanets.length; pi++) {
      const planet = hostPlanets[pi]
      const planetGroup = new THREE.Group()
      const orbitTilt = ((si * 7 + pi * 13) % 7 - 3) * 0.04
      const orbitGroup = new THREE.Group()
      orbitGroup.rotation.x = orbitTilt
      planetGroup.add(orbitGroup)

      const pivot = new THREE.Object3D()
      pivot.position.set(planet.distance, 0, 0)
      const mesh = new THREE.Mesh(SPHERE_GEO, planetMaterial(planet.type))
      mesh.scale.setScalar(planet.radius)
      pivot.add(mesh)
      orbitGroup.add(pivot)

      planetGroup.add(buildOrbitLine(planet.distance))
      starGroup.add(planetGroup)

      planetEntries.push({
        orbitGroup,
        mesh,
        orbitPeriod: planet.orbitPeriod,
        spinPeriod: planet.spinPeriod,
      })
    }

    // 该恒星的彗星
    const hostComets = data.comets.filter((c) => c.hostStarIndex === si)
    for (const comet of hostComets) {
      const cometGroup = new THREE.Group()
      const tex = glowTexture(64, 'rgba(255,255,255,1)', 'rgba(200,230,255,0.4)')
      const coreMat = new THREE.MeshBasicMaterial({
        map: tex,
        color: 0xffffff,
        toneMapped: false,
        transparent: true,
      })
      const core = new THREE.Mesh(COMET_NUCLEUS_GEO, coreMat)
      core.scale.setScalar(comet.nucleusRadius * 2)
      cometGroup.add(core)

      const tailGeo = new THREE.BufferGeometry()
      const positions = new Float32Array(comet.tailSegments * 3)
      tailGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3))
      const tailLine = new THREE.Line(tailGeo, COMET_TAIL_MAT)
      cometGroup.add(tailLine)
      starGroup.add(cometGroup)

      cometEntries.push({ comet, node: cometGroup, tailPositions: positions, tailLine, starPos })
    }
  }

  return {
    group,
    starsLocal,
    update(_dt: number, time: number) {
      // 行星公转 + 自转
      for (const p of planetEntries) {
        p.orbitGroup.rotation.y = ((time / p.orbitPeriod) * Math.PI * 2) % (Math.PI * 2)
        p.mesh.rotation.y += _dt * ((Math.PI * 2) / p.spinPeriod)
      }
      // 彗星 Kepler
      for (const c of cometEntries) {
        const n = (2 * Math.PI) / c.comet.orbitPeriod
        const M = c.comet.meanAnomaly0 + n * time
        const E = solveKepler(M, c.comet.eccentricity)
        const a = c.comet.semiMajorAxis
        const e = c.comet.eccentricity
        const x = a * (Math.cos(E) - e)
        const y = a * Math.sqrt(1 - e * e) * Math.sin(E)
        // 局部坐标（相对 starPos）
        c.node.position.set(x, 0, y)
        // 尾迹方向：从核心朝远离主星方向
        const len = Math.hypot(x, y) || 1
        const dirX = x / len
        const dirZ = y / len
        const tailLen = Math.min(8, a * 0.05)
        const segs = c.comet.tailSegments
        const positions = c.tailPositions
        for (let i = 0; i < segs; i++) {
          const t = i / (segs - 1)
          positions[i * 3 + 0] = x + dirX * t * tailLen
          positions[i * 3 + 1] = 0
          positions[i * 3 + 2] = y + dirZ * t * tailLen
        }
        ;(c.tailLine.geometry.attributes.position as THREE.BufferAttribute).needsUpdate = true
      }
    },
    dispose() {
      group.parent?.remove(group)
    },
  }
}