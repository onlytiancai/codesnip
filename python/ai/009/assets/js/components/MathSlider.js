// MathSlider.js — 通用数学互动（斜率 / 截距 / 矩阵元素）
// 用途：演示直线 y = mx + b
import { ref, onMounted } from "vue";

export const MathSlider = {
  props: ["data"],
  setup(props) {
    // 默认：直线 y = m·x + b
    const m = ref(1.0);
    const b = ref(0.0);
    const canvasRef = ref(null);

    function draw() {
      const c = canvasRef.value;
      if (!c) return;
      const ctx = c.getContext("2d");
      const W = c.width, H = c.height;
      ctx.clearRect(0, 0, W, H);

      const cs = getComputedStyle(document.documentElement);
      const accent = cs.getPropertyValue("--accent").trim();
      const muted = cs.getPropertyValue("--muted").trim();
      const text = cs.getPropertyValue("--text").trim();

      const cx = W / 2, cy = H / 2;
      const scale = 40;

      // 网格
      ctx.strokeStyle = muted + "22";
      ctx.lineWidth = 0.5;
      for (let i = -5; i <= 5; i++) {
        ctx.beginPath();
        ctx.moveTo(cx + i * scale, 0); ctx.lineTo(cx + i * scale, H); ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(0, cy + i * scale); ctx.lineTo(W, cy + i * scale); ctx.stroke();
      }

      // 坐标轴
      ctx.strokeStyle = muted + "88";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, cy); ctx.lineTo(W, cy);
      ctx.moveTo(cx, 0); ctx.lineTo(cx, H);
      ctx.stroke();

      // 直线
      ctx.strokeStyle = accent;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      const x1 = -6, y1 = m.value * x1 + b.value;
      const x2 = 6, y2 = m.value * x2 + b.value;
      const px1 = cx + x1 * scale, py1 = cy - y1 * scale;
      const px2 = cx + x2 * scale, py2 = cy - y2 * scale;
      ctx.moveTo(px1, py1);
      ctx.lineTo(px2, py2);
      ctx.stroke();

      // 截距标记
      if (Math.abs(b.value) < 5) {
        ctx.fillStyle = accent;
        ctx.beginPath();
        ctx.arc(cx, cy - b.value * scale, 5, 0, 2 * Math.PI);
        ctx.fill();
      }

      // 标签
      ctx.fillStyle = text;
      ctx.font = "12px ui-monospace, monospace";
      ctx.fillText(`y = ${m.value.toFixed(2)}·x + ${b.value.toFixed(2)}`, 10, 20);
      ctx.fillText(`斜率 m = ${m.value.toFixed(2)}`, 10, 38);
      ctx.fillText(`截距 b = ${b.value.toFixed(2)}`, 10, 56);
    }

    onMounted(draw);

    return { m, b, canvasRef, draw };
  },
  template: `
    <div style="margin:24px 0; padding:20px; background:var(--panel2); border-radius:12px">
      <h4 style="margin-top:0">📐 互动：拖动滑块看直线 y = mx + b</h4>
      <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(220px, 1fr)); gap:12px; margin:12px 0">
        <label>斜率 m = {{ m.toFixed(2) }}<input type="range" v-model.number="m" min="-3" max="3" step="0.1" /></label>
        <label>截距 b = {{ b.toFixed(2) }}<input type="range" v-model.number="b" min="-3" max="3" step="0.1" /></label>
      </div>
      <div style="background:var(--bg); border-radius:8px; padding:8px">
        <canvas ref="canvasRef" width="600" height="260" style="width:100%; height:auto; max-width:600px; display:block; margin:0 auto"></canvas>
      </div>
    </div>
  `,
  updated() { this.draw && this.draw(); },
};
