/**
 * 应用级图标 —— 直接用 **lucide-vue-next**（ISC 协议，https://lucide.dev）。
 *
 * 单独一个 .ts 文件：
 *  - 名字 → 组件的映射表放在这儿，`AppIcon.vue` 只管渲染
 *  - `AppIconName` 类型能被别人 import（`AppIcon.vue` 里 defineProps 用得上）
 */
import {
  BookOpen,
  CaseSensitive,
  ChevronDown,
  Languages,
  PanelLeftClose,
  PanelLeftOpen,
  PanelRightClose,
  PanelRightOpen,
  Power,
  Quote,
  Replace,
  Settings,
  SlidersHorizontal,
  TextAlignStart,
  Type,
  User,
  Zap,
} from 'lucide-vue-next'
import type { Component } from 'vue'

/** 所有可用图标的 name。 */
export type AppIconName =
  | 'languages'
  | 'case-sensitive'
  | 'panel-left-close'
  | 'panel-left-open'
  | 'panel-right-close'
  | 'panel-right-open'
  | 'user'
  | 'settings'
  | 'power'
  | 'chevron-down'
  // 右侧工具栏
  | 'sliders-horizontal'
  | 'zap'
  | 'type'
  | 'quote'
  | 'text-align-start'
  | 'book-open'
  | 'replace'

/** name → lucide 图标组件 */
export const APP_ICONS: Record<AppIconName, Component> = {
  // 文A —— 中译英
  languages: Languages,
  // Aa —— 英译中
  'case-sensitive': CaseSensitive,
  'panel-left-close': PanelLeftClose,
  'panel-left-open': PanelLeftOpen,
  'panel-right-close': PanelRightClose,
  'panel-right-open': PanelRightOpen,
  user: User,
  settings: Settings,
  power: Power,
  'chevron-down': ChevronDown,
  // 右侧工具栏
  'sliders-horizontal': SlidersHorizontal,
  zap: Zap,
  type: Type,
  quote: Quote,
  'text-align-start': TextAlignStart,
  'book-open': BookOpen,
  replace: Replace,
}
