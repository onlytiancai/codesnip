/**
 * 全局应用状态（自由飞行版）。
 */

import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react'

export type Backend = 'webgpu' | 'webgl2' | null
export type SpeedMode = 'normal' | 'boost'

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
  pointerLocked: boolean
}

export interface SceneApi {
  teleport(x: number, y: number, z: number): void
  requestPointerLock(): void
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
  pointerLocked: false,
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