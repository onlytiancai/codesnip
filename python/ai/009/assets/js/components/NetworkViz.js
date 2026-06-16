// NetworkViz.js — 可交互神经网络拓扑
// 鼠标悬停节点高亮，hover 看激活值（这里用静态展示版）
import { ref } from "vue";

export const NetworkViz = {
  props: ["data"],
  setup(props) {
    const hover = ref(null);

    // 默认 2-4-1 MLP
    const layers = [
      [{ id: "i0", label: "x₁", x: 80 }, { id: "i1", label: "x₂", x: 80 }],
      [
        { id: "h0", label: "h₁" }, { id: "h1", label: "h₂" },
        { id: "h2", label: "h₃" }, { id: "h3", label: "h₄" },
      ],
      [{ id: "o0", label: "ŷ" }],
    ];
    const layerXs = [80, 240, 400];
    const ys = [70, 130, 190];

    function nodePos(layerIdx, nodeIdx) {
      const x = layerXs[layerIdx];
      const layer = layers[layerIdx];
      const totalH = ys[layer.length - 1] - ys[0] || 1;
      const y = layer.length === 1
        ? 130
        : ys[0] + (nodeIdx / (layer.length - 1)) * totalH;
      return { x, y };
    }

    const edges = [];
    for (let li = 0; li < layers.length - 1; li++) {
      for (const a of layers[li]) {
        for (const b of layers[li + 1]) {
          edges.push({ from: a, to: b, fromLayer: li });
        }
      }
    }

    return { hover, layers, edges, nodePos, layerXs };
  },
  template: `
    <div style="margin:24px 0; padding:24px; background:var(--panel2); border-radius:12px; overflow-x:auto">
      <svg viewBox="0 0 480 260" xmlns="http://www.w3.org/2000/svg" style="width:100%; max-width:480px; height:auto">
        <defs>
          <marker id="arrow-net" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--muted)"/>
          </marker>
        </defs>

        <!-- 边 -->
        <g fill="none" stroke="var(--muted)" stroke-width="0.8" stroke-opacity="0.4">
          <line v-for="(e, i) in edges" :key="i"
                :x1="nodePos(e.fromLayer, layers[e.fromLayer].indexOf(e.from)).x"
                :y1="nodePos(e.fromLayer, layers[e.fromLayer].indexOf(e.from)).y"
                :x2="nodePos(e.fromLayer + 1, layers[e.fromLayer + 1].indexOf(e.to)).x"
                :y2="nodePos(e.fromLayer + 1, layers[e.fromLayer + 1].indexOf(e.to)).y"
                marker-end="url(#arrow-net)"/>
        </g>

        <!-- 节点 -->
        <g>
          <template v-for="(layer, li) in layers" :key="li">
            <g v-for="(n, ni) in layer" :key="n.id" style="cursor:pointer"
               @mouseenter="hover = n.id" @mouseleave="hover = null">
              <circle :cx="nodePos(li, ni).x" :cy="nodePos(li, ni).y" r="22"
                      :fill="hover === n.id ? 'var(--accent)' : 'var(--panel)'"
                      :stroke="li === 0 ? 'var(--accent2)' : (li === 2 ? 'var(--danger)' : 'var(--accent)')"
                      stroke-width="2"/>
              <text :x="nodePos(li, ni).x" :y="nodePos(li, ni).y" text-anchor="middle" dominant-baseline="central"
                    fill="currentColor" font-size="13" font-weight="600" font-family="ui-monospace, monospace">
                {{ n.label }}
              </text>
            </g>
          </template>
        </g>

        <!-- 层标签 -->
        <g fill="currentColor" font-size="11" opacity="0.7" text-anchor="middle">
          <text x="80" y="240">输入层 (2)</text>
          <text x="240" y="240">隐藏层 (4)</text>
          <text x="400" y="240">输出层 (1)</text>
        </g>
      </svg>
    </div>
  `,
};
