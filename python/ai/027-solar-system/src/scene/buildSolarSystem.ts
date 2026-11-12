/**
 * 太阳系场景构建：太阳（发光核心 + 光晕）、八大行星、月球、土星环、轨道线、CSS2D 标签。
 *
 * 结构（以地球为例）：
 *   scene
 *   ├── orbitGroup（rotation.y = 公转角）
 *   │   └── pivot（轨道位置，标签/相机跟随锚点）
 *   │       ├── tiltGroup（rotation.z = 轴倾角）
 *   │       │   ├── planetMesh（rotation.y = 自转角）
 *   │       │   ├── cloudMesh（地球云层，稍大球壳）
 *   │       │   └── ringMesh（土星/天王星环）
 *   │       └── moonOrbitGroup（rotation.y = 月球公转角）
 *   │           └── moonPivot → moonMesh
 *   ├── orbitLine（Line，独立于公转组，静态圆环）
 *   └── ...
 */

import * as THREE from 'three/webgpu'
import { CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js'
import { BODIES, MOON_DISTANCE_EARTH_KM, PLANET_ORDER, type BodyData, type BodyId } from '../data/planetData'
import {
  MOON_VISUAL_DISTANCE,
  SUN_VISUAL_RADIUS,
  visualDistance,
  visualOrbitSeconds,
  visualRadius,
  visualSpinSeconds,
} from '../utils/scale'
import { formatDistanceKm, formatPeriodDays } from '../utils/format'
import type { QualityProfile } from '../utils/performance'
import * as TEX from './ProceduralTextures'

export interface BuiltBody {
  id: BodyId
  data: BodyData
  /** 本体网格（拾取/高亮目标） */
  mesh: THREE.Mesh
  /** 云层（地球） */
  cloudMesh: THREE.Mesh | null
  /** 环（土星/天王星） */
  ringMesh: THREE.Mesh | null
  /** 轨道位置锚点（标签与相机跟随） */
  pivot: THREE.Object3D
  /** 公转组（太阳为 null） */
  orbitGroup: THREE.Object3D | null
  /** 轴倾角组 */
  tiltGroup: THREE.Object3D
  /** 轨道线（太阳/月球为 null）。用 Line + 闭合端点，WebGPURenderer 不支持 LineLoop */
  orbitLine: THREE.Line | null
  /** 月球绕地球小轨道（地球专属） */
  moonOrbitLine: THREE.Line | null
  /** 月球公转组（地球专属） */
  moonOrbitGroup: THREE.Object3D | null
  /** 视觉自转周期（秒，负 = 逆向） */
  spinSeconds: number
  /** 视觉公转周期（秒，0 = 不公转） */
  orbitSeconds: number
  /** 选中高亮用：基础自发光 */
  baseEmissive: THREE.Color
  /** 轨道模式下变暗的材质 */
  dimmable: THREE.Material[]
  /** CSS2D 标签 */
  label: CSS2DObject | null
  visible: boolean
}

export interface SolarSystemHandles {
  bodies: Map<BodyId, BuiltBody>
  sunGlow: THREE.Object3D[]
  guideGroup: THREE.Group
  orbitLines: THREE.Line[]
  labels: CSS2DObject[]
  sunLight: THREE.PointLight
  onLabelClick: (id: BodyId) => void
}

const GOLDEN_ANGLE = 2.399963 // 黄金角，让初始相位错开

export function buildSolarSystem(
  scene: THREE.Scene,
  renderer: THREE.WebGPURenderer,
  profile: QualityProfile,
  onLabelClick: (id: BodyId) => void,
): SolarSystemHandles {
  const bodies = new Map<BodyId, BuiltBody>()
  const labels: CSS2DObject[] = []
  const orbitLines: THREE.Line[] = []
  const sunGlow: THREE.Sprite[] = []

  const texW = profile.textureWidth
  const segments = profile.sphereSegments

  // ---------- 太阳 ----------
  const sunGeo = new THREE.SphereGeometry(SUN_VISUAL_RADIUS, segments, Math.max(16, segments / 2))
  const sunMat = new THREE.MeshBasicMaterial({ map: TEX.sunTexture(texW), toneMapped: false })
  const sunMesh = new THREE.Mesh(sunGeo, sunMat)
  scene.add(sunMesh)

  // 不加 sprite/球壳光晕 —— 这些会在近距离视角占满屏幕造成"模糊"
// 太阳的发光完全依赖 MeshBasicMaterial（toneMapped:false）的纯白 + bloom 的高阈值泛光
// 这样无论相机离太阳多近都不会糊屏（视差由 bloom 自然处理）
  sunGlow.length = 0

  // 灯光：太阳点光源（decay=0 避免外行星过暗）+ 适度环境光 + 程序化 IBL 环境贴图
  // 注意：envMapIntensity 由 meshStandardMaterial.envMapIntensity 控制，默认 1；
  // 这里通过 PMREM 烘焙低强度环境贴图，让暗面保留冷蓝细节但不过曝
  const sunLight = new THREE.PointLight(0xffffff, 3.2, 0, 0)
  scene.add(sunLight)
  scene.add(new THREE.AmbientLight(0x4a5e88, 0.35))
  // 程序化 IBL：从一张 256x256 渐变画布 → PMREM → Equirect 环境贴图
  // 渐变很弱（接近黑），仅让背阳面保留 5% 亮度，避免"全黑看不到细节"
  const pmrem = new THREE.PMREMGenerator(renderer)
  const envCanvas = document.createElement('canvas')
  envCanvas.width = envCanvas.height = 256
  const ectx = envCanvas.getContext('2d')!
  const grad = ectx.createLinearGradient(0, 0, 0, 256)
  grad.addColorStop(0, '#06080f')
  grad.addColorStop(0.5, '#0a1028')
  grad.addColorStop(1, '#101a3a')
  ectx.fillStyle = grad
  ectx.fillRect(0, 0, 256, 256)
  const envTex = new THREE.CanvasTexture(envCanvas)
  envTex.mapping = THREE.EquirectangularReflectionMapping
  const envMap = pmrem.fromEquirectangular(envTex).texture
  scene.environment = envMap
  scene.environmentIntensity = 0.4 // r185 支持：全局环境贴图强度，控制在低位避免洗白
  pmrem.dispose()
  envTex.dispose()

  const sunBody: BuiltBody = {
    id: 'sun',
    data: BODIES.sun,
    mesh: sunMesh,
    cloudMesh: null,
    ringMesh: null,
    pivot: sunMesh,
    orbitGroup: null,
    tiltGroup: sunMesh,
    orbitLine: null,
    moonOrbitLine: null,
    moonOrbitGroup: null,
    spinSeconds: visualSpinSeconds(BODIES.sun.rotationPeriodHours),
    orbitSeconds: 0,
    baseEmissive: new THREE.Color(0, 0, 0),
    dimmable: [sunMat],
    label: null,
    visible: true,
  }
  bodies.set('sun', sunBody)

  // ---------- 八大行星 ----------
  let phase = 0
  for (const id of PLANET_ORDER) {
    const data = BODIES[id]
    const dist = visualDistance(data.distanceAu!)
    const r = visualRadius(data.radiusKm)

    const orbitGroup = new THREE.Group()
    orbitGroup.rotation.y = phase
    orbitGroup.userData.phase = phase
    scene.add(orbitGroup)
    phase += GOLDEN_ANGLE

    const pivot = new THREE.Group()
    pivot.position.set(dist, 0, 0)
    orbitGroup.add(pivot)

    const tiltGroup = new THREE.Group()
    tiltGroup.rotation.z = THREE.MathUtils.degToRad(data.axialTiltDeg)
    pivot.add(tiltGroup)

    // ---- 行星本体 ----
    // 球壳分段数按视角距离自动调整：近距离需要更多面避免边缘呈多边形
    const sphereSeg = profile.sphereSegments
    const geo = new THREE.SphereGeometry(r, sphereSeg, Math.max(24, sphereSeg / 2))
    const mat = new THREE.MeshStandardMaterial({
      map: planetTexture(id, texW, profile.textureWidthLarge),
      roughness: 0.9,
      metalness: 0,
      // emissive 必须为 0：彩色 emissive 会被 bloom 阈值（0.85）拾取，整个行星被泛光糊掉
      emissive: new THREE.Color(0x000000),
      emissiveIntensity: 0,
    })
    const mesh = new THREE.Mesh(geo, mat)
    mesh.rotation.y = phase * 3
    tiltGroup.add(mesh)

    const entry: BuiltBody = {
      id,
      data,
      mesh,
      cloudMesh: null,
      ringMesh: null,
      pivot,
      orbitGroup,
      tiltGroup,
      orbitLine: null,
      moonOrbitLine: null,
      moonOrbitGroup: null,
      spinSeconds: visualSpinSeconds(data.rotationPeriodHours),
      orbitSeconds: visualOrbitSeconds(data.orbitalPeriodDays),
      baseEmissive: new THREE.Color(data.accent).multiplyScalar(0.07),
      dimmable: [mat],
      label: null,
      visible: true,
    }

    // ---- 地球：云层 + 月球 ----
    if (id === 'earth' && profile.cloudLayer) {
      // 云层：略大球壳 + 真透明纹理（不透明覆盖 = 把地球糊掉）+ 无环境反射
      const cloudGeo = new THREE.SphereGeometry(r * 1.012, segments, Math.max(16, segments / 2))
      const cloudMat = new THREE.MeshBasicMaterial({
        map: TEX.earthCloudTexture(profile.textureWidthLarge),
        transparent: true,
        depthWrite: false,
        opacity: 0.78,
        blending: THREE.NormalBlending,
      })
      const clouds = new THREE.Mesh(cloudGeo, cloudMat)
      clouds.renderOrder = 1
      tiltGroup.add(clouds)
      entry.cloudMesh = clouds
      entry.dimmable.push(cloudMat)
    }
    if (id === 'earth') {
      // 月球绕地球公转
      const moonOrbitGroup = new THREE.Group()
      pivot.add(moonOrbitGroup)
      entry.moonOrbitGroup = moonOrbitGroup

      const moonData = BODIES.moon
      const moonR = visualRadius(moonData.radiusKm)
      const moonPivot = new THREE.Group()
      moonPivot.position.set(MOON_VISUAL_DISTANCE, 0, 0)
      moonOrbitGroup.add(moonPivot)
      const moonGeo = new THREE.SphereGeometry(moonR, segments, Math.max(12, segments / 2))
      const moonMat = new THREE.MeshStandardMaterial({
        map: TEX.moonTexture(texW),
        roughness: 1,
        metalness: 0,
        emissive: new THREE.Color(0x888888).multiplyScalar(0.06),
      })
      const moonMesh = new THREE.Mesh(moonGeo, moonMat)
      moonPivot.add(moonMesh)
      const moonEntry: BuiltBody = {
        id: 'moon',
        data: moonData,
        mesh: moonMesh,
        cloudMesh: null,
        ringMesh: null,
        pivot: moonPivot,
        orbitGroup: null,
        tiltGroup: moonPivot,
        orbitLine: null,
        moonOrbitLine: null,
        moonOrbitGroup: null,
        spinSeconds: visualSpinSeconds(moonData.rotationPeriodHours),
        orbitSeconds: visualOrbitSeconds(moonData.orbitalPeriodDays),
        baseEmissive: new THREE.Color(0x888888).multiplyScalar(0.06),
        dimmable: [moonMat],
        label: null,
        visible: true,
      }
      bodies.set('moon', moonEntry)

      // 月球小轨道（地球 → 月球关系提示）
      const moonOrbit = makeOrbitLine(MOON_VISUAL_DISTANCE, 96, 0xffffff, 0.16)
      pivot.add(moonOrbit)
      entry.moonOrbitLine = moonOrbit

      // 月球标签
      const moonLabel = makeLabel('moon', moonData, moonR)
      moonPivot.add(moonLabel)
      moonEntry.label = moonLabel
      labels.push(moonLabel)
    }

    // ---- 土星环 / 天王星环 ----
    if (id === 'saturn' || id === 'uranus') {
      const inner = id === 'saturn' ? r * 1.28 : r * 1.55
      const outer = id === 'saturn' ? r * 2.45 : r * 1.95
      const ringGeo = new THREE.RingGeometry(inner, outer, 160, 1)
      const ringTexW = id === 'saturn' ? profile.textureWidthLarge : texW
      const ringMat = new THREE.MeshStandardMaterial({
        map: id === 'saturn' ? TEX.saturnRingTexture(ringTexW) : TEX.uranusRingTexture(ringTexW),
        transparent: true,
        side: THREE.DoubleSide,
        depthWrite: false,
        roughness: 1,
        metalness: 0,
        emissive: new THREE.Color(0xffffff).multiplyScalar(id === 'saturn' ? 0.06 : 0.04),
      })
      const ring = new THREE.Mesh(ringGeo, ringMat)
      ring.rotation.x = -Math.PI / 2 // 环面与行星赤道面（XZ）重合
      tiltGroup.add(ring)
      entry.ringMesh = ring
      entry.dimmable.push(ringMat)
    }

    // ---- 轨道线 ----
    const orbitLine = makeOrbitLine(dist, profile.orbitPoints, 0x7f96c8, 0.32)
    scene.add(orbitLine)
    entry.orbitLine = orbitLine
    orbitLines.push(orbitLine)

    // ---- 标签 ----
    const label = makeLabel(id, data, r)
    pivot.add(label)
    entry.label = label
    labels.push(label)

    bodies.set(id, entry)
  }

  // ---------- 太阳标签 ----------
  const sunLabel = makeLabel('sun', BODIES.sun, SUN_VISUAL_RADIUS)
  sunMesh.add(sunLabel)
  sunBody.label = sunLabel
  labels.push(sunLabel)

  // ---------- 辅助参考线（默认隐藏） ----------
  const guideGroup = new THREE.Group()
  const grid = new THREE.GridHelper(100, 40, 0x335577, 0x223344)
  const gridMats = Array.isArray(grid.material) ? grid.material : [grid.material]
  for (const gm of gridMats) {
    gm.transparent = true
    gm.opacity = 0.22
    gm.depthWrite = false
  }
  guideGroup.add(grid)
  // 轨道平面大圆
  const guideRing = makeOrbitLine(45, 200, 0x4477aa, 0.3)
  guideGroup.add(guideRing)
  guideGroup.visible = false
  scene.add(guideGroup)

  // 标签点击 → 选中
  for (const label of labels) {
    label.element.addEventListener('pointerdown', (e) => {
      e.stopPropagation()
      onLabelClick(label.userData.bodyId as BodyId)
    })
  }

  return { bodies, sunGlow, guideGroup, orbitLines, labels, sunLight, onLabelClick }
}

// ---------- 内部工具 ----------

/** 大行星用更高分辨率纹理（近距离观察更清晰）；texW=常规宽，largeW=大纹理宽 */
function planetTexture(id: BodyId, w: number, largeW: number): THREE.CanvasTexture {
  const useLarge = id === 'earth' || id === 'jupiter' || id === 'saturn'
  const size = useLarge ? largeW : w
  switch (id) {
    case 'mercury':
      return TEX.mercuryTexture(size)
    case 'venus':
      return TEX.venusTexture(size)
    case 'earth':
      return TEX.earthTexture(size)
    case 'mars':
      return TEX.marsTexture(size)
    case 'jupiter':
      return TEX.jupiterTexture(size)
    case 'saturn':
      return TEX.saturnTexture(size)
    case 'uranus':
      return TEX.uranusTexture(size)
    case 'neptune':
      return TEX.neptuneTexture(size)
    default:
      return TEX.moonTexture(size)
  }
}

function makeOrbitLine(radius: number, points: number, color: number, opacity: number): THREE.Line {
  const geo = new THREE.BufferGeometry()
  // 首尾重复一个点，实现闭合圆环（WebGPURenderer 不支持 LineLoop）
  const positions = new Float32Array((points + 1) * 3)
  for (let i = 0; i <= points; i++) {
    const a = (i / points) * Math.PI * 2
    positions[i * 3] = Math.cos(a) * radius
    positions[i * 3 + 1] = 0
    positions[i * 3 + 2] = Math.sin(a) * radius
  }
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3))
  const mat = new THREE.LineBasicMaterial({
    color,
    transparent: true,
    opacity,
    depthWrite: false,
  })
  return new THREE.Line(geo, mat)
}

function makeLabel(id: BodyId, data: BodyData, radius: number): CSS2DObject {
  const el = document.createElement('div')
  el.className = 'planet-label'
  el.setAttribute('role', 'button')
  el.setAttribute('aria-label', `查看${data.nameZh}`)
  const name = document.createElement('span')
  name.className = 'planet-label-name'
  name.textContent = `${data.emoji} ${data.nameZh}`
  const dist = document.createElement('span')
  dist.className = 'planet-label-sub'
  dist.hidden = true
  const period = document.createElement('span')
  period.className = 'planet-label-sub'
  period.hidden = true
  if (data.orbitAround === 'sun' && data.distanceFromSunMkm) {
    dist.textContent = `距太阳 ${formatDistanceKm(data.distanceFromSunMkm * 1e6)}`
    period.textContent = `公转 ${formatPeriodDays(data.orbitalPeriodDays)}`
  } else if (id === 'moon') {
    dist.textContent = `距地球 ${formatDistanceKm(MOON_DISTANCE_EARTH_KM)}`
    period.textContent = `绕地球 ${formatPeriodDays(data.orbitalPeriodDays)}`
  }
  el.append(name, dist, period)
  const label = new CSS2DObject(el)
  label.position.set(0, radius + 0.9, 0)
  label.userData.bodyId = id
  label.userData.subDistEl = dist
  label.userData.subPeriodEl = period
  return label
}
