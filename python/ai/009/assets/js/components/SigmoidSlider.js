// SigmoidSlider.js — 互动 sigmoid 滑块
// 拖动 w/b 看 σ(z) 输出变化
import { ref, computed, onMounted, watch } from "vue";

export const SigmoidSlider = {
  props: ["data"],
  setup(props) {
    const w = ref(1.0);
    const x = ref(1.0);
    const b = ref(0.0);
    const canvasRef = ref(null);

    const z = computed(() => w.value * x.value + b.value);
    const y = computed(() => 1 / (1 + Math.exp(-z.value)));

    function draw() {
      const c = canvasRef.value;
      if (!c) return;
      const ctx = c.getContext("2d");
      const W = c.width, H = c.height;
      ctx.clearRect(0, 0, W, H);

      // 读取 CSS 变量颜色
      const cs = getComputedStyle(document.documentElement);
      const accent = cs.getPropertyValue("--accent").trim();
      const accent2 = cs.getPropertyValue("--accent2").trim();
      const muted = cs.getPropertyValue("--muted").trim();
      const text = cs.getPropertyValue("--text").trim();

      // 坐标轴
      const cx = W / 2, cy = H / 2;
      const scale = 40;
      ctx.strokeStyle = muted + "66";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, cy); ctx.lineTo(W, cy);
      ctx.moveTo(cx, 0); ctx.lineTo(cx, H);
      ctx.stroke();

      // 网格
      ctx.strokeStyle = muted + "22";
      ctx.lineWidth = 0.5;
      for (let i = -5; i <= 5; i++) {
        ctx.beginPath();
        ctx.moveTo(cx + i * scale, 0); ctx.lineTo(cx + i * scale, H); ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(0, cy + i * scale); ctx.lineTo(W, cy + i * scale); ctx.stroke();
      }

      // sigmoid 曲线
      ctx.strokeStyle = accent;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      for (let px = 0; px <= W; px++) {
        const zVal = (px - cx) / scale;
        const yVal = 1 / (1 + Math.exp(-zVal));
        const py = cy - (yVal - 0.5) * scale * 4;
        if (px === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.stroke();

      // 当前点
      const currentX = cx + z.value * scale;
      const currentY = cy - (y.value - 0.5) * scale * 4;
      ctx.fillStyle = accent2;
      ctx.beginPath();
      ctx.arc(currentX, currentY, 6, 0, 2 * Math.PI);
      ctx.fill();
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 2;
      ctx.stroke();

      // 标签
      ctx.fillStyle = text;
      ctx.font = "12px ui-monospace, monospace";
      ctx.fillText(`z = ${w.value}·${x.value.toFixed(2)} + ${b.value} = ${z.value.toFixed(3)}`, 10, 20);
      ctx.fillText(`σ(z) = ${y.value.toFixed(4)}`, 10, 38);
    }

    onMounted(() => {
      draw();
    });

    watch([w, x, b], () => { draw(); });

    return { w, x, b, z, y, canvasRef, draw };
  },
  template: `
    <div style="margin:24px 0; padding:20px; background:var(--panel2); border-radius:12px">
      <h4 style="margin-top:0">🧪 互动实验：拖动滑块看 σ(wx + b) 输出</h4>
      <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(180px, 1fr)); gap:12px; margin:12px 0">
        <label>权重 w = {{ w.toFixed(2) }}<input type="range" v-model.number="w" min="-3" max="3" step="0.1" /></label>
        <label>输入 x = {{ x.toFixed(2) }}<input type="range" v-model.number="x" min="-3" max="3" step="0.1" /></label>
        <label>偏置 b = {{ b.toFixed(2) }}<input type="range" v-model.number="b" min="-3" max="3" step="0.1" /></label>
      </div>
      <div style="background:var(--bg); border-radius:8px; padding:8px">
        <canvas ref="canvasRef" width="600" height="260" style="width:100%; height:auto; max-width:600px; display:block; margin:0 auto"></canvas>
      </div>
    </div>
  `,
  updated() {
    this.draw && this.draw();
  },
};
