/**
 * 全局应用状态：Context + useReducer + localStorage 持久化（仅设置项）。
 * 3D 场景的指令式操作（flyTo / resetView 等）通过 SceneApi 接口转发，
 * 避免 React 渲染循环与 three 渲染循环互相干扰。
 */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useReducer,
  useRef,
  type ReactNode,
} from 'react'
import type { BodyId } from '../data/planetData'
import { DEFAULT_TIME_SCALE } from '../utils/scale'

export type ViewMode = 'explore' | 'science' | 'orbits'
export type QualitySetting = 'auto' | 'high' | 'low'
export type Backend = 'webgpu' | 'webgl2' | null

export interface AppSettings {
  // 显示
  showOrbits: boolean
  showLabels: boolean
  showData: boolean // 名称标签下的数据小字（距离/周期）
  showStarfield: boolean
  showGuide: boolean // 辅助参考线
  // 动画
  autoRotate: boolean
  orbitEnabled: boolean // 公转
  spinEnabled: boolean // 自转
  // 视觉
  bloomEnabled: boolean
  shadowsEnabled: boolean
  quality: QualitySetting
  // 科普
  showRealData: boolean // 信息面板真实数据
  showFacts: boolean // 小知识
  showDistances: boolean // 标签上的距离
  showPeriods: boolean // 标签上的周期
  // 音效（默认关闭）
  soundEnabled: boolean
}

export const DEFAULT_SETTINGS: AppSettings = {
  showOrbits: true,
  showLabels: false,
  showData: false,
  showStarfield: true,
  showGuide: false,
  autoRotate: false,
  orbitEnabled: true,
  spinEnabled: true,
  bloomEnabled: false, // 默认关闭：bloom 后处理在某些驱动下会产生大模糊半径泛光，覆盖屏幕导致"模糊"
  shadowsEnabled: false,
  quality: 'auto',
  showRealData: true,
  showFacts: true,
  showDistances: false,
  showPeriods: false,
  soundEnabled: false,
}

export interface AppState {
  settings: AppSettings
  timeScale: number
  paused: boolean
  selectedId: BodyId | null
  viewMode: ViewMode
  backend: Backend
  degraded: boolean
  ready: boolean
  fatalError: string | null
  resolvedQuality: 'high' | 'low'
  welcomeDone: boolean
  settingsOpen: boolean
}

const initialState: AppState = {
  settings: DEFAULT_SETTINGS,
  timeScale: DEFAULT_TIME_SCALE,
  paused: false,
  selectedId: null,
  viewMode: 'explore',
  backend: null,
  degraded: false,
  ready: false,
  fatalError: null,
  resolvedQuality: 'high',
  welcomeDone: false,
  settingsOpen: false,
}

type Action =
  | { type: 'SET_SETTING'; key: keyof AppSettings; value: boolean | QualitySetting }
  | { type: 'SET_TIME_SCALE'; value: number }
  | { type: 'TOGGLE_PAUSED' }
  | { type: 'SELECT_BODY'; id: BodyId | null }
  | { type: 'SET_VIEW_MODE'; mode: ViewMode }
  | { type: 'SET_BACKEND'; backend: Backend; degraded: boolean }
  | { type: 'SET_READY' }
  | { type: 'SET_FATAL'; message: string }
  | { type: 'SET_RESOLVED_QUALITY'; quality: 'high' | 'low' }
  | { type: 'DISMISS_WELCOME' }
  | { type: 'SET_SETTINGS_OPEN'; open: boolean }

const STORAGE_KEY = 'solar-system-settings-v1'

function loadSettings(): AppSettings {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (raw) return { ...DEFAULT_SETTINGS, ...JSON.parse(raw) }
  } catch {
    /* 隐私模式等场景下忽略 */
  }
  return DEFAULT_SETTINGS
}

function reducer(state: AppState, action: Action): AppState {
  switch (action.type) {
    case 'SET_SETTING':
      return {
        ...state,
        settings: { ...state.settings, [action.key]: action.value },
      }
    case 'SET_TIME_SCALE':
      return { ...state, timeScale: action.value }
    case 'TOGGLE_PAUSED':
      return { ...state, paused: !state.paused }
    case 'SELECT_BODY':
      return { ...state, selectedId: action.id }
    case 'SET_VIEW_MODE':
      return { ...state, viewMode: action.mode }
    case 'SET_BACKEND':
      return { ...state, backend: action.backend, degraded: action.degraded }
    case 'SET_READY':
      return { ...state, ready: true }
    case 'SET_FATAL':
      return { ...state, fatalError: action.message }
    case 'SET_RESOLVED_QUALITY':
      return { ...state, resolvedQuality: action.quality }
    case 'DISMISS_WELCOME':
      return { ...state, welcomeDone: true }
    case 'SET_SETTINGS_OPEN':
      return { ...state, settingsOpen: action.open }
    default:
      return state
  }
}

/** 3D 场景对 UI 暴露的指令式接口（由 SceneManager 实现） */
export interface SceneApi {
  selectBody(id: BodyId | null, opts?: { fly?: boolean }): void
  resetView(): void
  setTimeScale(v: number): void
  setPaused(p: boolean): void
  applySettings(s: AppSettings): void
  setViewMode(m: ViewMode): void
  setResolvedQuality(q: 'high' | 'low'): void
}

interface AppContextValue {
  state: AppState
  dispatch: React.Dispatch<Action>
  sceneRef: React.RefObject<SceneApi | null>
  /** UI 主动飞向天体 */
  flyTo(id: BodyId): void
  /** UI 点击天体（选中 + 飞行） */
  selectBody(id: BodyId): void
}

const AppContext = createContext<AppContextValue | null>(null)

export function AppStateProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, initialState, (s) => ({
    ...s,
    settings: loadSettings(),
  }))
  const sceneRef = useRef<SceneApi | null>(null)

  // 设置持久化
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state.settings))
    } catch {
      /* ignore */
    }
  }, [state.settings])

  // 设置变化 → 场景
  useEffect(() => {
    sceneRef.current?.applySettings(state.settings)
  }, [state.settings])

  // 时间速度 / 暂停 → 场景
  useEffect(() => {
    sceneRef.current?.setTimeScale(state.timeScale)
  }, [state.timeScale])
  useEffect(() => {
    sceneRef.current?.setPaused(state.paused)
  }, [state.paused])

  // 显示模式 → 场景
  useEffect(() => {
    sceneRef.current?.setViewMode(state.viewMode)
  }, [state.viewMode])

  const flyTo = useCallback((id: BodyId) => {
    sceneRef.current?.selectBody(id, { fly: true })
  }, [])

  const selectBody = useCallback((id: BodyId) => {
    sceneRef.current?.selectBody(id, { fly: true })
  }, [])

  const value = useMemo(
    () => ({ state, dispatch, sceneRef, flyTo, selectBody }),
    [state, flyTo, selectBody],
  )

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>
}

export function useApp(): AppContextValue {
  const ctx = useContext(AppContext)
  if (!ctx) throw new Error('useApp must be used within AppStateProvider')
  return ctx
}
