// ComputeGraph.js — SVG 计算图（X → W1 → z1 → σ → a1 → W2 → z2 → σ → ŷ）
// 静态展示版（不需要交互）
import { computed } from "vue";

export const ComputeGraph = {
  props: ["data"],
  setup(props) {
    // 默认展示 2-4-1 MLP 的前向计算图
    const nodes = [
      { id: "x1", label: "x₁", x: 60, y: 60, type: "input" },
      { id: "x2", label: "x₂", x: 60, y: 140, type: "input" },
      { id: "h1", label: "h₁", x: 200, y: 40, type: "hidden" },
      { id: "h2", label: "h₂", x: 200, y: 90, type: "hidden" },
      { id: "h3", label: "h₃", x: 200, y: 140, type: "hidden" },
      { id: "h4", label: "h₄", x: 200, y: 190, type: "hidden" },
      { id: "a1_1", label: "a₁¹", x: 320, y: 40, type: "act" },
      { id: "a1_2", label: "a₁²", x: 320, y: 90, type: "act" },
      { id: "a1_3", label: "a₁³", x: 320, y: 140, type: "act" },
      { id: "a1_4", label: "a₁⁴", x: 320, y: 190, type: "act" },
      { id: "y", label: "ŷ", x: 460, y: 115, type: "output" },
    ];

    // 边：x → h（4 节点），h → a（identity），a → y
    const edges = [];
    for (const inp of ["x1", "x2"]) {
      for (const hid of ["h1", "h2", "h3", "h4"]) {
        edges.push({ from: inp, to: hid, type: "weight" });
      }
    }
    for (const hid of ["h1", "h2", "h3", "h4"]) {
      const a = "a1_" + hid.slice(1);
      edges.push({ from: hid, to: a, type: "sigma" });
      edges.push({ from: a, to: "y", type: "weight" });
    }

    function nodeById(id) { return nodes.find((n) => n.id === id); }

    const edgePaths = computed(() => edges.map((e) => {
      const a = nodeById(e.from), b = nodeById(e.to);
      return {
        d: `M ${a.x} ${a.y} C ${(a.x + b.x) / 2} ${a.y}, ${(a.x + b.x) / 2} ${b.y}, ${b.x} ${b.y}`,
        type: e.type,
      };
    }));

    return { nodes, edgePaths };
  },
  template: `
    <div style="margin:24px 0; padding:20px; background:var(--panel2); border-radius:12px; overflow-x:auto">
      <svg viewBox="0 0 540 240" xmlns="http://www.w3.org/2000/svg" style="width:100%; max-width:540px; height:auto">
        <defs>
          <marker id="arrow-cg" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor"/>
          </marker>
        </defs>

        <!-- 边 -->
        <g fill="none" stroke-width="1.4" stroke-opacity="0.6">
          <path v-for="(e, i) in edgePaths" :key="i" :d="e.d" :stroke="e.type === 'sigma' ? 'var(--warn)' : 'var(--accent2)'"
                :stroke-dasharray="e.type === 'sigma' ? '4 3' : ''" marker-end="url(#arrow-cg)" color="var(--text)"/>
        </g>

        <!-- 节点 -->
        <g>
          <g v-for="n in nodes" :key="n.id">
            <circle :cx="n.x" :cy="n.y" r="18" :fill="
              n.type === 'input' ? 'var(--accent2)' :
              n.type === 'hidden' ? 'var(--accent)' :
              n.type === 'act' ? 'var(--warn)' : 'var(--danger)'
            " fill-opacity="0.18" :stroke="
              n.type === 'input' ? 'var(--accent2)' :
              n.type === 'hidden' ? 'var(--accent)' :
              n.type === 'act' ? 'var(--warn)' : 'var(--danger)'
            " stroke-width="1.5"/>
            <text :x="n.x" :y="n.y" text-anchor="middle" dominant-baseline="central"
                  fill="currentColor" font-size="13" font-weight="600" font-family="ui-monospace, monospace">
              {{ n.label }}
            </text>
          </g>
        </g>

        <!-- 标签 -->
        <text x="290" y="20" text-anchor="middle" fill="currentColor" font-size="11" opacity="0.6">前向传播 (Forward Pass)</text>
      </svg>
      <div style="text-align:center; margin-top:8px; font-size:12px; color:var(--muted)">
        实线箭头 = 矩阵乘法 W·x；虚线箭头 = 激活函数 σ
      </div>
    </div>
  `,
};
