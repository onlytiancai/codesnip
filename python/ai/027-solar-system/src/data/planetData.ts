/**
 * 真实天文数据层（数据来源：NASA Planetary Fact Sheet https://nssdc.gsfc.nasa.gov/planetary/factsheet/）
 * 注意：这里只存真实数据；视觉展示比例一律在 utils/scale.ts 中换算，两套体系严格分离。
 */

export type BodyType = 'star' | 'rocky' | 'gasGiant' | 'iceGiant' | 'moon'

export type BodyId =
  | 'sun'
  | 'mercury'
  | 'venus'
  | 'earth'
  | 'mars'
  | 'jupiter'
  | 'saturn'
  | 'uranus'
  | 'neptune'
  | 'moon'

export interface BodyData {
  id: BodyId
  nameZh: string
  nameEn: string
  emoji: string
  type: BodyType
  /** 半径 km */
  radiusKm: number
  /** 距太阳平均距离 AU（月球没有） */
  distanceAu?: number
  /** 距太阳平均距离 百万 km（月球没有） */
  distanceFromSunMkm?: number
  /** 公转周期（天）——月球为绕地球周期 */
  orbitalPeriodDays: number
  /** 自转周期（小时），负值表示逆向自转（金星、天王星） */
  rotationPeriodHours: number
  /** 自转轴倾角（度） */
  axialTiltDeg: number
  /** 卫星数量 */
  moons: number
  /** 表面/平均温度 ℃ */
  surfaceTempC?: number
  /** 绕什么公转 */
  orbitAround: 'sun' | 'earth'
  /** UI 强调色 */
  accent: string
}

export const BODIES: Record<BodyId, BodyData> = {
  sun: {
    id: 'sun',
    nameZh: '太阳',
    nameEn: 'Sun',
    emoji: '☀️',
    type: 'star',
    radiusKm: 696_340,
    orbitalPeriodDays: 0,
    rotationPeriodHours: 609.12, // 太阳自转周期约 25.38 天（赤道）
    axialTiltDeg: 7.25,
    moons: 8, // 八大行星都绕它转
    surfaceTempC: 5505,
    orbitAround: 'sun',
    accent: '#ffb340',
  },
  mercury: {
    id: 'mercury',
    nameZh: '水星',
    nameEn: 'Mercury',
    emoji: '☿️',
    type: 'rocky',
    radiusKm: 2439.7,
    distanceAu: 0.387,
    distanceFromSunMkm: 57.9,
    orbitalPeriodDays: 87.97,
    rotationPeriodHours: 1407.6,
    axialTiltDeg: 0.03,
    moons: 0,
    surfaceTempC: 167,
    orbitAround: 'sun',
    accent: '#9e9e9e',
  },
  venus: {
    id: 'venus',
    nameZh: '金星',
    nameEn: 'Venus',
    emoji: '♀️',
    type: 'rocky',
    radiusKm: 6051.8,
    distanceAu: 0.723,
    distanceFromSunMkm: 108.2,
    orbitalPeriodDays: 224.7,
    rotationPeriodHours: -5832.5, // 逆向自转
    axialTiltDeg: 177.4,
    moons: 0,
    surfaceTempC: 464,
    orbitAround: 'sun',
    accent: '#ffd77a',
  },
  earth: {
    id: 'earth',
    nameZh: '地球',
    nameEn: 'Earth',
    emoji: '🌍',
    type: 'rocky',
    radiusKm: 6371,
    distanceAu: 1,
    distanceFromSunMkm: 149.6,
    orbitalPeriodDays: 365.25,
    rotationPeriodHours: 23.9,
    axialTiltDeg: 23.4,
    moons: 1,
    surfaceTempC: 15,
    orbitAround: 'sun',
    accent: '#5ab3ff',
  },
  mars: {
    id: 'mars',
    nameZh: '火星',
    nameEn: 'Mars',
    emoji: '♂️',
    type: 'rocky',
    radiusKm: 3389.5,
    distanceAu: 1.524,
    distanceFromSunMkm: 227.9,
    orbitalPeriodDays: 687,
    rotationPeriodHours: 24.6,
    axialTiltDeg: 25.2,
    moons: 2,
    surfaceTempC: -65,
    orbitAround: 'sun',
    accent: '#ff8a5c',
  },
  jupiter: {
    id: 'jupiter',
    nameZh: '木星',
    nameEn: 'Jupiter',
    emoji: '♃',
    type: 'gasGiant',
    radiusKm: 69911,
    distanceAu: 5.203,
    distanceFromSunMkm: 778.5,
    orbitalPeriodDays: 4331,
    rotationPeriodHours: 9.9,
    axialTiltDeg: 3.1,
    moons: 95,
    surfaceTempC: -110,
    orbitAround: 'sun',
    accent: '#e8b48c',
  },
  saturn: {
    id: 'saturn',
    nameZh: '土星',
    nameEn: 'Saturn',
    emoji: '🪐',
    type: 'gasGiant',
    radiusKm: 58232,
    distanceAu: 9.537,
    distanceFromSunMkm: 1434,
    orbitalPeriodDays: 10747,
    rotationPeriodHours: 10.7,
    axialTiltDeg: 26.7,
    moons: 274,
    surfaceTempC: -140,
    orbitAround: 'sun',
    accent: '#f2d9a8',
  },
  uranus: {
    id: 'uranus',
    nameZh: '天王星',
    nameEn: 'Uranus',
    emoji: '♅',
    type: 'iceGiant',
    radiusKm: 25362,
    distanceAu: 19.19,
    distanceFromSunMkm: 2871,
    orbitalPeriodDays: 30589,
    rotationPeriodHours: -17.2, // 逆向自转
    axialTiltDeg: 97.8, // 躺着转
    moons: 28,
    surfaceTempC: -195,
    orbitAround: 'sun',
    accent: '#8fdcff',
  },
  neptune: {
    id: 'neptune',
    nameZh: '海王星',
    nameEn: 'Neptune',
    emoji: '♆',
    type: 'iceGiant',
    radiusKm: 24622,
    distanceAu: 30.07,
    distanceFromSunMkm: 4495,
    orbitalPeriodDays: 59800,
    rotationPeriodHours: 16.1,
    axialTiltDeg: 28.3,
    moons: 16,
    surfaceTempC: -200,
    orbitAround: 'sun',
    accent: '#5d7bff',
  },
  moon: {
    id: 'moon',
    nameZh: '月球',
    nameEn: 'Moon',
    emoji: '🌙',
    type: 'moon',
    radiusKm: 1737.4,
    orbitalPeriodDays: 27.32, // 绕地球
    rotationPeriodHours: 655.7, // 潮汐锁定，自转=公转
    axialTiltDeg: 6.7,
    moons: 0,
    surfaceTempC: -20,
    orbitAround: 'earth',
    accent: '#d8d8d8',
  },
}

/** 月球与地球平均距离 384,400 km（用于信息面板） */
export const MOON_DISTANCE_EARTH_KM = 384_400

/** 太阳年龄（约 46 亿年） */
export const SUN_AGE_BILLION_YEARS = 4.6

/** 八大行星顺序（由内到外） */
export const PLANET_ORDER: BodyId[] = [
  'mercury',
  'venus',
  'earth',
  'mars',
  'jupiter',
  'saturn',
  'uranus',
  'neptune',
]

/** 导航列表顺序 */
export const NAV_ORDER: BodyId[] = ['sun', ...PLANET_ORDER, 'moon']
