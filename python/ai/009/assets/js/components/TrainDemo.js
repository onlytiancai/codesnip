// TrainDemo.js — 嵌 chart.js 的迷你训练演示
// 用于 ch09 展示损失下降曲线（按"开始训练"按钮触发）
import { ref, onMounted, onBeforeUnmount } from "vue";

export const TrainDemo = {
  props: ["data"],
  setup(props) {
    const epochs = ref(200);
    const lr = ref(1.0);
    const running = ref(false);
    const chartRef = ref(null);
    const finalAcc = ref(0);
    const finalLoss = ref(0);
    let chart = null;
    let timer = null;

    function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }

    function train() {
      if (running.value) return;
      running.value = true;
      // 简单的 XOR 2-4-1 训练模拟（固定 seed）
      const N = epochs.value;
      const X = [[0,0],[0,1],[1,0],[1,1]];
      const Y = [0,1,1,0];

      // 初始化（固定）
      const W1 = [[1.5, -1.0, 0.5, 1.2], [-1.0, 1.5, 1.2, 0.5]];
      const b1 = [-0.5, 0.3, -0.2, 0.1];
      const W2 = [1.0, 1.0, 1.0, 1.0];
      const b2 = -1.5;

      const losses = [];
      const accs = [];
      const labels = [];

      let i = 0;
      const stepMs = 50;
      timer = setInterval(() => {
        if (i >= N) { running.value = false; clearInterval(timer); return; }
        // 1 个 epoch：4 个样本
        let loss = 0, correct = 0;
        for (let s = 0; s < 4; s++) {
          const x = X[s], y = Y[s];
          const z1 = [W1[0][0]*x[0]+W1[1][0]*x[1]+b1[0], W1[0][1]*x[0]+W1[1][1]*x[1]+b1[1], W1[0][2]*x[0]+W1[1][2]*x[1]+b1[2], W1[0][3]*x[0]+W1[1][3]*x[1]+b1[3]];
          const a1 = z1.map(sigmoid);
          const z2 = a1[0]*W2[0]+a1[1]*W2[1]+a1[2]*W2[2]+a1[3]*W2[3]+b2;
          const yhat = sigmoid(z2);
          const err = y - yhat;
          loss += -(y * Math.log(yhat + 1e-9) + (1-y) * Math.log(1-yhat + 1e-9));
          if ((yhat > 0.5 ? 1 : 0) === y) correct++;
          // 简化反向更新（demo 用）
          for (let k = 0; k < 4; k++) W2[k] += lr.value * err * a1[k] * 0.1;
          b2 += lr.value * err * 0.1;
        }
        losses.push(loss / 4);
        accs.push(correct / 4);
        labels.push(i + 1);

        if (chart) {
          chart.data.labels = labels;
          chart.data.datasets[0].data = losses;
          chart.data.datasets[1].data = accs;
          chart.update("none");
        }
        finalLoss.value = losses[losses.length - 1];
        finalAcc.value = accs[accs.length - 1];
        i++;
      }, stepMs);
    }

    function reset() {
      if (timer) clearInterval(timer);
      running.value = false;
      if (chart) {
        chart.data.labels = [];
        chart.data.datasets[0].data = [];
        chart.data.datasets[1].data = [];
        chart.update();
      }
      finalLoss.value = 0;
      finalAcc.value = 0;
    }

    onMounted(() => {
      if (!window.Chart) {
        console.warn("Chart.js not loaded");
        return;
      }
      const ctx = chartRef.value.getContext("2d");
      const accent = getComputedStyle(document.documentElement).getPropertyValue("--accent").trim();
      const accent2 = getComputedStyle(document.documentElement).getPropertyValue("--accent2").trim();
      const text = getComputedStyle(document.documentElement).getPropertyValue("--text").trim();
      const muted = getComputedStyle(document.documentElement).getPropertyValue("--muted").trim();

      chart = new window.Chart(ctx, {
        type: "line",
        data: {
          labels: [],
          datasets: [
            { label: "Loss", data: [], borderColor: accent, backgroundColor: accent + "33", tension: 0.3, yAxisID: "y" },
            { label: "Accuracy", data: [], borderColor: accent2, backgroundColor: accent2 + "33", tension: 0.3, yAxisID: "y1" },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          animation: false,
          scales: {
            x: { ticks: { color: muted }, grid: { color: muted + "22" } },
            y: { type: "linear", position: "left", ticks: { color: muted }, grid: { color: muted + "22" }, title: { display: true, text: "Loss", color: muted } },
            y1: { type: "linear", position: "right", min: 0, max: 1, ticks: { color: muted }, grid: { display: false }, title: { display: true, text: "Accuracy", color: muted } },
          },
          plugins: {
            legend: { labels: { color: text } },
          },
        },
      });
    });

    onBeforeUnmount(() => {
      if (timer) clearInterval(timer);
      if (chart) chart.destroy();
    });

    return { epochs, lr, running, chartRef, finalAcc, finalLoss, train, reset };
  },
  template: `
    <div style="margin:24px 0; padding:20px; background:var(--panel2); border-radius:12px">
      <h4 style="margin-top:0">🚀 训练演示：2-4-1 MLP 解决 XOR</h4>

      <div style="display:flex; gap:16px; margin:12px 0; flex-wrap:wrap; align-items:center">
        <label style="display:flex; align-items:center; gap:6px; font-size:13px">
          <span>学习率 lr</span>
          <input type="number" v-model.number="lr" min="0.01" max="5" step="0.1" style="width:70px" />
        </label>
        <label style="display:flex; align-items:center; gap:6px; font-size:13px">
          <span>训练步数</span>
          <input type="number" v-model.number="epochs" min="10" max="1000" step="10" style="width:80px" />
        </label>
        <button class="primary" @click="train" :disabled="running">
          {{ running ? '⏳ 训练中...' : '▶ 开始训练' }}
        </button>
        <button @click="reset" :disabled="running">重置</button>
      </div>

      <div style="height:280px; position:relative; background:var(--bg); border-radius:8px; padding:8px">
        <canvas ref="chartRef"></canvas>
      </div>

      <div v-if="finalLoss > 0" style="display:flex; gap:24px; margin-top:12px; font-size:13px">
        <div>最终 Loss: <strong style="color:var(--accent)">{{ finalLoss.toFixed(4) }}</strong></div>
        <div>最终 Acc: <strong style="color:var(--accent2)">{{ (finalAcc * 100).toFixed(0) }}%</strong></div>
      </div>
    </div>
  `,
};
