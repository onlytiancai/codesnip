/**
 * 全局应用状态（自由飞行版）。
 *
 * escapeState：
 *   'playing' — pointer locked，飞行中
 *   'paused'  — Esc 后第一次：显示「序列化星空」提示框 + pointer unlocked
 *   'menu'    — Esc 后第二次：关闭提示框，pointer 仍未 lock
 */

import {
  createContext,
  useContext,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react'

export type Backend = 'webgpu' | 'webgl2' | null
export type SpeedMode = 'normal' | 'boost'
export type EscapeState = 'playing' | 'paused' | 'menu'

export interface Vec3Like {
  x: number
  y: number
  z: number
}

export interface AppState {
  ready: boolean
  fatalError: string | null
  backend: Backend
  degraded: boolean
  position: Vec3Like
  speedMode: SpeedMode
  chunkCount: number
  escapeState: EscapeState
}

export interface SystemLabel {
  id: number
  distance: number
  screenX: number
  screenY: number
  inFront: boolean
}

export interface SceneApi {
  teleport(x: number, y: number, z: number): void
  requestPointerLock(): void
  getNearbyLabels(maxRadius: number): SystemLabel[]
}

interface AppContextValue {
  state: AppState
  setState: React.Dispatch<React.SetStateAction<AppState>>
  sceneApiRef: React.RefObject<SceneApi | null>
}

const AppContext = createContext<AppContextValue | null>(null)

const INITIAL_STATE: AppState = {
  ready: false,
  fatalError: null,
  backend: null,
  degraded: false,
  position: { x: 0, y: 0, z: 0 },
  speedMode: 'normal',
  chunkCount: 0,
  escapeState: 'menu',
}

export function AppStateProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<AppState>(INITIAL_STATE)
  const sceneApiRef = useRef<SceneApi | null>(null)
  const value = useMemo(() => ({ state, setState, sceneApiRef }), [state])
  return <AppContext.Provider value={value}>{children}</AppContext.Provider>
}

export function useApp(): AppContextValue {
  const ctx = useContext(AppContext)
  if (!ctx) throw new Error('useApp must be used within AppStateProvider')
  return ctx
}