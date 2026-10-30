/**
 * 选区菜单 / 工具栏按钮能触发的动作。
 * 单列一份，菜单项和工具栏按钮都从这里取，避免两边漂移。
 *
 * 注意：'lookup'（查词）是**本地词库**查询，不调模型 —— 两个方向都有。
 */
export type AiAction =
  | 'candidates'
  | 'tone'
  | 'improve'
  | 'shorten'
  | 'expand'
  | 'grammar'
  | 'instruction'
  | 'lookup'
