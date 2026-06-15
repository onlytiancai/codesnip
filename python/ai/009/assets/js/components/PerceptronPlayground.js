// PerceptronPlayground.js — 交互式感知机演示
// ch04 用：拖动 w1, w2, b 滑块，看决策直线 w1·x1 + w2·x2 + b = 0 怎么切 4 个点
// AND / OR 模式可解（4/4），XOR 模式永远 ≤ 3/4（线性不可分）
import { ref, watch, onMounted, onBeforeUnmount, computed } from "vue";
import { useStore } from "../store.js";
import { I18N, pick } from "../i18n.js";

const DEFAULTS = {
  AND: { w1: 1.0, w2: 1.0, b: -1.5 },
  OR:  { w1: 1.0, w2: 1.0, b: -0.5 },
  XOR: { w1: 1.0, w2: 1.0, b: -0.5 },
};
const TARGETS = { AND: [0, 0, 0, 1], OR: [0, 1, 1, 1], XOR: [0, 1, 1, 0] };
const POINTS  = [[0, 0], [0, 1], [1, 0], [1, 1]];

const CSS_SIZE = 360;          // 画布 CSS 像素尺寸
const XMIN = -0.5;
const XMAX = 1.5;
const PAD = 24;

export const PerceptronPlayground = {
  props: ["data"],
  setup(props) {
    const { state } = useStore();
    const t = (k) => pick(I18N.ui[k] || { zh: k, en: k }, state.language);

    // 响应式状态
    const mode = ref("AND");
    const w1   = ref(DEFAULTS.AND.w1);
    const w2   = ref(DEFAULTS.AND.w2);
    const b    = ref(DEFAULTS.AND.b);

    const canvasRef   = ref(null);
    const predictions = ref([0, 0, 0, 0]);
    const zs          = ref([0, 0, 0, 0]);
    const target      = computed(() => TARGETS[mode.value]);
    const accuracy    = computed(() =>
      predictions.value.reduce(
        (s, p, i) => s + (p === TARGETS[mode.value][i] ? 1 : 0),
        0
      )
    );

    function setMode(m) {
      mode.value = m;
      w1.value = DEFAULTS[m].w1;
      w2.value = DEFAULTS[m].w2;
      b.value  = DEFAULTS[m].b;
    }

    function recompute() {
      zs.value = POINTS.map(([x1, x2]) => w1.value * x1 + w2.value * x2 + b.value);
      predictions.value = zs.value.map((v) => (v > 0 ? 1 : 0));
    }

    function draw() {
      const canvas = canvasRef.value;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      // sx/sy 走逻辑尺寸（与 CSS 一致），与 devicePixelRatio 缩放无关
      const sx = (v) => PAD + ((v - XMIN) / (XMAX - XMIN)) * (CSS_SIZE - 2 * PAD);
      const sy = (v) => CSS_SIZE - PAD - ((v - XMIN) / (XMAX - XMIN)) * (CSS_SIZE - 2 * PAD);

      // 读 CSS 变量（主题切换时 CSS 变量已变，每次 draw 都能拿到最新色）
      const cs = getComputedStyle(document.documentElement);
      const accent  = cs.getPropertyValue("--accent").trim();
      const accent2 = cs.getPropertyValue("--accent2").trim();
      const danger  = cs.getPropertyValue("--danger").trim();
      const muted   = cs.getPropertyValue("--muted").trim();
      const text    = cs.getPropertyValue("--text").trim();

      ctx.clearRect(0, 0, CSS_SIZE, CSS_SIZE);

      // 网格（x1 / x2 ∈ {0, 0.5, 1}）
      ctx.strokeStyle = muted + "22";
      ctx.lineWidth = 1;
      for (const v of [0, 0.5, 1]) {
        ctx.beginPath();
        ctx.moveTo(sx(v), PAD); ctx.lineTo(sx(v), CSS_SIZE - PAD);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(PAD, sy(v)); ctx.lineTo(CSS_SIZE - PAD, sy(v));
        ctx.stroke();
      }
      // 坐标轴（穿过 0 的两条粗线）
      ctx.strokeStyle = muted + "66";
      ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.moveTo(sx(0), PAD); ctx.lineTo(sx(0), CSS_SIZE - PAD); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(PAD, sy(0)); ctx.lineTo(CSS_SIZE - PAD, sy(0)); ctx.stroke();

      // 决策直线：w1·x1 + w2·x2 + b = 0  ⇒  x2 = -(w1·x1 + b) / w2
      ctx.save();
      ctx.strokeStyle = accent2;
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
      ctx.beginPath();
      if (Math.abs(w2.value) < 1e-6) {
        // w2 ≈ 0：退化为垂直线 x1 = -b/w1
        const xv = -b.value / w1.value;
        if (xv >= XMIN && xv <= XMAX) {
          ctx.moveTo(sx(xv), PAD);
          ctx.lineTo(sx(xv), CSS_SIZE - PAD);
        }
      } else {
        const y0 = -(w1.value * XMIN + b.value) / w2.value;
        const y1 = -(w1.value * XMAX + b.value) / w2.value;
        ctx.moveTo(sx(XMIN), sy(y0));
        ctx.lineTo(sx(XMAX), sy(y1));
      }
      ctx.stroke();
      ctx.restore();

      // 4 个点
      const tgt = TARGETS[mode.value];
      POINTS.forEach(([x1, x2], i) => {
        const correct = predictions.value[i] === tgt[i];
        const color = correct ? accent : danger;
        // halo（正确=绿、错误=红的半透明圆）
        ctx.beginPath();
        ctx.arc(sx(x1), sy(x2), 14, 0, Math.PI * 2);
        ctx.fillStyle = color + "22";
        ctx.fill();
        // 圆（白心 + 彩边）
        ctx.beginPath();
        ctx.arc(sx(x1), sy(x2), 9, 0, Math.PI * 2);
        ctx.fillStyle = "#fff";
        ctx.fill();
        ctx.lineWidth = 2.5;
        ctx.strokeStyle = color;
        ctx.stroke();
        // ŷ 标签
        ctx.fillStyle = text;
        ctx.font = "11px ui-monospace, monospace";
        ctx.textAlign = "center";
        ctx.fillText("ŷ=" + predictions.value[i], sx(x1), sy(x2) - 18);
      });

      // 轴标签
      ctx.fillStyle = muted;
      ctx.font = "12px ui-sans-serif, system-ui";
      ctx.textAlign = "right";
      ctx.fillText("x₁ →", CSS_SIZE - PAD, sy(0) + 14);
      ctx.textAlign = "left";
      ctx.fillText("x₂ ↑", sx(0) + 6, PAD + 12);
    }

    function redraw() {
      recompute();
      draw();
    }

    watch([w1, w2, b, mode], redraw);

    let themeObserver = null;
    onMounted(() => {
      const canvas = canvasRef.value;
      // 高 DPI 适配：让线条在 Retina 屏不模糊
      const dpr = window.devicePixelRatio || 1;
      canvas.width = CSS_SIZE * dpr;
      canvas.height = CSS_SIZE * dpr;
      canvas.style.width = CSS_SIZE + "px";
      canvas.style.height = CSS_SIZE + "px";
      const ctx = canvas.getContext("2d");
      ctx.scale(dpr, dpr);
      redraw();
      // 主题切换时 CSS 变量已更新，但滑块空闲不触发 watch；
      // 用 MutationObserver 强制重绘以应用新配色
      themeObserver = new MutationObserver(redraw);
      themeObserver.observe(document.documentElement, {
        attributes: true,
        attributeFilter: ["data-theme"],
      });
    });

    onBeforeUnmount(() => {
      if (themeObserver) themeObserver.disconnect();
    });

    return { t, mode, w1, w2, b, canvasRef, predictions, zs, accuracy, target, setMode };
  },
  template: `
    <div style="margin:24px 0; padding:20px; background:var(--panel2); border-radius:12px">
      <h4 style="margin:0 0 6px">{{ t('ppTitle') }}</h4>
      <p style="margin:0 0 14px; color:var(--muted); font-size:13px">{{ t('ppSubtitle') }}</p>

      <div style="display:grid; grid-template-columns:minmax(300px,360px) 1fr; gap:20px; align-items:start">
        <!-- 画布 -->
        <div style="background:var(--bg); border-radius:8px; padding:6px; display:flex; justify-content:center">
          <canvas ref="canvasRef" style="border-radius:6px"></canvas>
        </div>

        <!-- 控制面板 -->
        <div>
          <div style="display:flex; gap:8px; margin-bottom:14px; align-items:center; flex-wrap:wrap">
            <span style="font-size:13px; color:var(--muted)">{{ t('ppMode') }}:</span>
            <button v-for="m in ['AND','OR','XOR']" :key="m"
                    :class="mode === m ? 'primary' : ''" @click="setMode(m)">{{ m }}</button>
          </div>

          <label style="display:grid; grid-template-columns:90px 1fr 50px; gap:8px; align-items:center; font-size:13px; margin-bottom:8px">
            <span style="font-family:ui-monospace,monospace">{{ t('ppW1') }}</span>
            <input type="range" min="-2" max="2" step="0.1" v-model.number="w1" />
            <span style="font-family:ui-monospace,monospace; text-align:right; color:var(--muted)">{{ w1.toFixed(1) }}</span>
          </label>
          <label style="display:grid; grid-template-columns:90px 1fr 50px; gap:8px; align-items:center; font-size:13px; margin-bottom:8px">
            <span style="font-family:ui-monospace,monospace">{{ t('ppW2') }}</span>
            <input type="range" min="-2" max="2" step="0.1" v-model.number="w2" />
            <span style="font-family:ui-monospace,monospace; text-align:right; color:var(--muted)">{{ w2.toFixed(1) }}</span>
          </label>
          <label style="display:grid; grid-template-columns:90px 1fr 50px; gap:8px; align-items:center; font-size:13px; margin-bottom:8px">
            <span style="font-family:ui-monospace,monospace">{{ t('ppBias') }}</span>
            <input type="range" min="-2" max="2" step="0.1" v-model.number="b" />
            <span style="font-family:ui-monospace,monospace; text-align:right; color:var(--muted)">{{ b.toFixed(1) }}</span>
          </label>

          <button @click="setMode(mode)" style="margin-top:6px">{{ t('ppReset') }}</button>
        </div>
      </div>

      <!-- 正确率 -->
      <div style="margin-top:16px; display:flex; align-items:baseline; gap:12px">
        <div :style="'font-size:42px; font-weight:700; font-family:ui-monospace,monospace; color:' + (accuracy === 4 ? 'var(--accent)' : accuracy >= 2 ? 'var(--warn)' : 'var(--danger)')">
          {{ accuracy }}<span style="font-size:24px; color:var(--muted)">/4</span>
        </div>
        <div style="color:var(--muted); font-size:13px">{{ t('ppAccuracy') }}</div>
      </div>

      <p v-if="mode === 'XOR' && accuracy < 4" style="margin:8px 0 0; font-size:12px; color:var(--danger)">
        {{ t('ppNoSolution') }}
      </p>

      <!-- 预测表 -->
      <table style="margin-top:14px; width:100%; border-collapse:collapse; font-size:13px; font-family:ui-monospace,monospace">
        <thead>
          <tr style="color:var(--muted); text-align:left">
            <th style="padding:4px 8px; border-bottom:1px solid var(--border)">{{ t('ppTableInput') }}</th>
            <th style="padding:4px 8px; border-bottom:1px solid var(--border)">{{ t('ppTableTarget') }}</th>
            <th style="padding:4px 8px; border-bottom:1px solid var(--border)">{{ t('ppTableZ') }}</th>
            <th style="padding:4px 8px; border-bottom:1px solid var(--border)">{{ t('ppTablePred') }}</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(pt, i) in [[0,0],[0,1],[1,0],[1,1]]" :key="i">
            <td style="padding:4px 8px">({{ pt[0] }}, {{ pt[1] }})</td>
            <td style="padding:4px 8px">{{ target[i] }}</td>
            <td style="padding:4px 8px; color:var(--muted)">{{ zs[i].toFixed(2) }}</td>
            <td :style="'padding:4px 8px; color:' + (predictions[i] === target[i] ? 'var(--accent)' : 'var(--danger)')">
              {{ predictions[i] }}
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  `,
};
